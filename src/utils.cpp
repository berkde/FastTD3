#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>

namespace fast_td3 {

torch::Tensor cpu_state(const torch::Tensor& tensor) {
    return tensor.detach().cpu();
}

torch::Tensor clamp_tensor(torch::Tensor tensor, float min_val, float max_val) {
    return torch::clamp(tensor, min_val, max_val);
}

void save_params(
    int global_step,
    std::shared_ptr<Actor> actor,
    std::shared_ptr<Critic> critic,
    std::shared_ptr<Critic> critic_target,
    std::shared_ptr<EmpiricalNormalization> obs_normalizer,
    std::shared_ptr<EmpiricalNormalization> critic_obs_normalizer,
    const std::string& save_path
) {
    torch::save(actor, save_path + "/actor_" + std::to_string(global_step) + ".pt");
    torch::save(critic, save_path + "/critic_" + std::to_string(global_step) + ".pt");
    torch::save(critic_target, save_path + "/critic_target_" + std::to_string(global_step) + ".pt");
    
    if (obs_normalizer) {
        torch::save(obs_normalizer, save_path + "/obs_normalizer_" + std::to_string(global_step) + ".pt");
    }
    if (critic_obs_normalizer) {
        torch::save(critic_obs_normalizer, save_path + "/critic_obs_normalizer_" + std::to_string(global_step) + ".pt");
    }
    
    spdlog::info("Saved parameters at step {} to {}", global_step, save_path);
}

void load_params(
    std::shared_ptr<Actor> actor,
    std::shared_ptr<Critic> critic,
    std::shared_ptr<Critic> critic_target,
    std::shared_ptr<EmpiricalNormalization> obs_normalizer,
    std::shared_ptr<EmpiricalNormalization> critic_obs_normalizer,
    const std::string& load_path
) {
    try {
        torch::load(actor, load_path + "/actor.pt");
        torch::load(critic, load_path + "/critic.pt");
        torch::load(critic_target, load_path + "/critic_target.pt");
        
        if (obs_normalizer) {
            torch::load(obs_normalizer, load_path + "/obs_normalizer.pt");
        }
        if (critic_obs_normalizer) {
            torch::load(critic_obs_normalizer, load_path + "/critic_obs_normalizer.pt");
        }
        
        spdlog::info("Loaded parameters from {}", load_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load parameters from {}: {}", load_path, e.what());
    }
}

torch::Tensor compute_target_q(
    std::shared_ptr<Critic> critic_target,
    torch::Tensor next_obs,
    torch::Tensor next_actions,
    torch::Tensor rewards,
    torch::Tensor dones,
    float gamma,
    bool use_cdq
) {
    auto [q1_proj, q2_proj] = critic_target->projection(next_obs, next_actions, rewards, 
                                                       torch::ones_like(dones), gamma);
    
    auto q1_values = critic_target->get_value(torch::softmax(q1_proj, 1));
    auto q2_values = critic_target->get_value(torch::softmax(q2_proj, 1));
    
    if (use_cdq) {
        return torch::min(q1_values, q2_values);
    } else {
        return q1_values;
    }
}

torch::Tensor compute_actor_loss(
    std::shared_ptr<Actor> actor,
    std::shared_ptr<Critic> critic,
    torch::Tensor obs,
    float policy_noise,
    float noise_clip
) {
    auto actions = actor->forward(obs);
    
    if (policy_noise > 0) {
        auto noise = torch::randn_like(actions) * policy_noise;
        noise = torch::clamp(noise, -noise_clip, noise_clip);
        actions = torch::clamp(actions + noise, -1.0, 1.0);
    }
    
    auto [q1, q2] = critic->forward(obs, actions);
    auto q1_values = critic->get_value(torch::softmax(q1, 1));
    auto q2_values = critic->get_value(torch::softmax(q2, 1));
    
    return -torch::min(q1_values, q2_values).mean();
}

torch::Tensor compute_critic_loss(
    std::shared_ptr<Critic> critic,
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor target_q,
    bool use_cdq
) {
    auto [q1, q2] = critic->forward(obs, actions);
    auto q1_values = critic->get_value(torch::softmax(q1, 1));
    auto q2_values = critic->get_value(torch::softmax(q2, 1));
    
    auto loss1 = torch::mse_loss(q1_values, target_q);
    auto loss2 = torch::mse_loss(q2_values, target_q);
    
    if (use_cdq) {
        return loss1 + loss2;
    } else {
        return loss1;
    }
}

torch::Device get_device(bool use_cuda, int device_rank) {
    if (use_cuda && torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, device_rank);
    } else {
        return torch::Device(torch::kCPU);
    }
}

std::string get_device_string(torch::Device device) {
    if (device.is_cuda()) {
        return "cuda:" + std::to_string(device.index());
    } else if (device.is_mps()) {
        return "mps:" + std::to_string(device.index());
    } else {
        return "cpu";
    }
}

void log_training_stats(
    int step,
    float actor_loss,
    float critic_loss,
    float total_reward,
    float episode_length,
    const std::string& log_file
) {
    std::stringstream ss;
    ss << "Step: " << step 
       << " | Actor Loss: " << actor_loss 
       << " | Critic Loss: " << critic_loss
       << " | Total Reward: " << total_reward
       << " | Episode Length: " << episode_length;
    
    spdlog::info(ss.str());
    
    if (!log_file.empty()) {
        std::ofstream file(log_file, std::ios::app);
        if (file.is_open()) {
            file << step << "," << actor_loss << "," << critic_loss 
                 << "," << total_reward << "," << episode_length << std::endl;
        }
    }
}

} // namespace fast_td3 