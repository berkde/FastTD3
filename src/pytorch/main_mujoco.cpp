#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "config.hpp"
#include "networks.hpp"
#include "replay_buffer.hpp"
#include "normalizers.hpp"
#include "utils.hpp"
#include "mujoco_env.hpp"

int main(int argc, char* argv[]) {
    // Setup logging
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("fast_td3_mujoco", console_sink);
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    
    // Parse configuration
    auto config = fast_td3::ConfigManager::parse_args(argc, argv);
    spdlog::info("Configuration loaded: env={}, agent={}, seed={}", 
                 config.env_name, config.agent, config.seed);
    
    // Set random seeds
    std::srand(config.seed);
    torch::manual_seed(config.seed);
    
    // Setup device
    auto device = fast_td3::get_device(config.cuda, config.device_rank);
    spdlog::info("Using device: {}", fast_td3::get_device_string(device));
    
    // Setup MuJoCo environment
    std::unique_ptr<fast_td3::MuJoCoEnvironment> env;
    try {
        env = fast_td3::create_mujoco_env(config.env_name, config.num_envs, device);
        spdlog::info("MuJoCo environment created successfully: {}", env->get_env_name());
    } catch (const std::exception& e) {
        spdlog::error("Failed to create MuJoCo environment: {}", e.what());
        spdlog::info("Available environments:");
        spdlog::info("  - {}", fast_td3::envs::HUMANOID_V5);
        spdlog::info("  - {}", fast_td3::envs::HUMANOID_STANDUP_V5);
        spdlog::info("  - {}", fast_td3::envs::WALKER_2D_V5);
        spdlog::info("  - {}", fast_td3::envs::HALF_CHEETAH_V5);
        spdlog::info("  - {}", fast_td3::envs::ANT_V5);
        spdlog::info("  - {}", fast_td3::envs::HOPPER_V5);
        spdlog::info("  - {}", fast_td3::envs::HUMANOID_V2);
        spdlog::info("  - {}", fast_td3::envs::WALKER_2D_V2);
        return 1;
    }
    
    int obs_dim = env->get_obs_dim();
    int act_dim = env->get_act_dim();
    
    spdlog::info("Environment dimensions: obs_dim={}, act_dim={}", obs_dim, act_dim);
    
    // Setup networks
    auto actor = std::make_shared<fast_td3::Actor>(
        obs_dim, act_dim, config.num_envs, config.init_scale, 
        config.actor_hidden_dim, config.std_min, config.std_max, device);
    
    auto critic = std::make_shared<fast_td3::Critic>(
        obs_dim, act_dim, config.num_atoms, config.v_min, config.v_max, 
        config.critic_hidden_dim, device);
    
    auto critic_target = std::make_shared<fast_td3::Critic>(
        obs_dim, act_dim, config.num_atoms, config.v_min, config.v_max, 
        config.critic_hidden_dim, device);
    
    // Manually copy parameters from critic to critic_target
    auto critic_params = critic->named_parameters();
    auto target_params = critic_target->named_parameters();
    for (const auto& item : critic_params) {
        if (target_params.contains(item.key())) {
            target_params[item.key()].copy_(item.value());
        }
    }
    
    // Setup optimizers
    auto actor_optimizer = torch::optim::Adam(
        actor->parameters(), 
        torch::optim::AdamOptions(config.actor_learning_rate).weight_decay(config.weight_decay)
    );
    
    auto critic_optimizer = torch::optim::Adam(
        critic->parameters(), 
        torch::optim::AdamOptions(config.critic_learning_rate).weight_decay(config.weight_decay)
    );
    
    // Setup normalizers
    std::shared_ptr<fast_td3::EmpiricalNormalization> obs_normalizer = nullptr;
    std::shared_ptr<fast_td3::EmpiricalNormalization> critic_obs_normalizer = nullptr;
    std::shared_ptr<fast_td3::RewardNormalizer> reward_normalizer = nullptr;
    
    if (config.obs_normalization) {
        obs_normalizer = std::make_shared<fast_td3::EmpiricalNormalization>(
            std::vector<int64_t>{obs_dim}, device);
        critic_obs_normalizer = std::make_shared<fast_td3::EmpiricalNormalization>(
            std::vector<int64_t>{obs_dim}, device);
    }
    
    if (config.reward_normalization) {
        reward_normalizer = std::make_shared<fast_td3::RewardNormalizer>(
            config.gamma, device);
    }
    
    // Setup replay buffer
    auto replay_buffer = std::make_unique<fast_td3::SimpleReplayBuffer>(
        config.num_envs, config.buffer_size, obs_dim, act_dim, obs_dim,
        false, false, config.num_steps, config.gamma, device);
    
    // Training loop
    int global_step = 0;
    int update_step = 0;
    float episode_reward = 0.0f;
    int episode_length = 0;
    
    spdlog::info("Starting MuJoCo training for {} timesteps", config.total_timesteps);
    
    // Reset environment
    env->reset();
    
    while (global_step < config.total_timesteps) {
        // Environment interaction
        auto obs = env->get_observations();
        
        // Normalize observations
        if (obs_normalizer) {
            obs_normalizer->update(obs);
            obs = obs_normalizer->forward(obs);
        }
        
        // Select actions
        torch::Tensor actions;
        if (global_step < config.learning_starts) {
            // Random actions during exploration
            actions = torch::rand({config.num_envs, act_dim}, device) * 2 - 1;
        } else {
            actions = actor->explore(obs);
        }
        
        // Environment step
        auto [next_obs, rewards, dones, truncations] = env->step(actions);
        
        // Normalize next observations
        if (obs_normalizer) {
            obs_normalizer->update(next_obs);
            next_obs = obs_normalizer->forward(next_obs);
        }
        
        // Normalize rewards
        if (reward_normalizer) {
            reward_normalizer->update_stats(rewards, dones);
            rewards = reward_normalizer->forward(rewards);
        }
        
        // Store transition
        fast_td3::Transition transition;
        transition.observations = obs;
        transition.actions = actions;
        transition.rewards = rewards;
        transition.dones = dones;
        transition.truncations = truncations;
        transition.next_observations = next_obs;
        transition.critic_observations = obs;
        transition.next_critic_observations = next_obs;
        
        replay_buffer->extend(transition);
        
        // Update episode statistics
        episode_reward += rewards.mean().item<float>();
        episode_length++;
        
        // Reset environment if done
        if (dones.any().item<bool>()) {
            env->reset();
            episode_reward = 0.0f;
            episode_length = 0;
        }
        
        // Training updates
        if (global_step >= config.learning_starts && 
            global_step % config.policy_frequency == 0) {
            
            for (int i = 0; i < config.num_updates; i++) {
                // Sample batch
                auto batch = replay_buffer->sample(config.batch_size / config.num_envs);
                
                // Normalize critic observations
                if (critic_obs_normalizer) {
                    critic_obs_normalizer->update(batch.critic_observations);
                    batch.critic_observations = critic_obs_normalizer->forward(batch.critic_observations);
                    batch.next_critic_observations = critic_obs_normalizer->forward(batch.next_critic_observations);
                }
                
                // Compute target Q values
                auto next_actions = actor->forward(batch.next_observations);
                auto target_q = fast_td3::compute_target_q(
                    critic_target, batch.next_critic_observations, next_actions,
                    batch.rewards, batch.dones, config.gamma, config.use_cdq);
                
                // Update critic
                critic_optimizer.zero_grad();
                auto critic_loss = fast_td3::compute_critic_loss(
                    critic, batch.critic_observations, batch.actions, target_q, config.use_cdq);
                critic_loss.backward({}, /*retain_graph=*/true);
                if (config.max_grad_norm > 0) {
                    torch::nn::utils::clip_grad_norm_(critic->parameters(), config.max_grad_norm);
                }
                critic_optimizer.step();
                
                // Update actor
                if (i % config.policy_frequency == 0) {
                    actor_optimizer.zero_grad();
                    auto actor_loss = fast_td3::compute_actor_loss(
                        actor, critic, batch.observations, config.policy_noise, config.noise_clip);
                    actor_loss.backward();
                    if (config.max_grad_norm > 0) {
                        torch::nn::utils::clip_grad_norm_(actor->parameters(), config.max_grad_norm);
                    }
                    actor_optimizer.step();
                    
                    // Update target network
                    torch::NoGradGuard no_grad;
                    for (auto& target_param : critic_target->parameters()) {
                        auto& param = critic->parameters()[&target_param - &critic_target->parameters()[0]];
                        target_param.copy_(config.tau * param + (1 - config.tau) * target_param);
                    }
                }
                
                update_step++;
            }
        }
        
        // Logging
        if (global_step % config.eval_interval == 0) {
            float avg_reward = rewards.mean().item<float>();
            float avg_episode_length = 1.0f / (dones.to(torch::kFloat).mean().item<float>() + 1e-8f);
            
            spdlog::info("Step: {} | Avg Reward: {:.3f} | Avg Episode Length: {:.1f} | Updates: {}", 
                        global_step, avg_reward, avg_episode_length, update_step);
        }
        
        // Save checkpoints
        if (global_step % config.save_interval == 0 && global_step > 0) {
            std::string save_path = "checkpoints/" + config.exp_name + "_" + std::to_string(config.seed);
            fast_td3::save_params(global_step, actor, critic, critic_target, 
                                 obs_normalizer, critic_obs_normalizer, save_path);
        }
        
        global_step += config.num_envs;
    }
    
    spdlog::info("MuJoCo training completed!");
    return 0;
} 