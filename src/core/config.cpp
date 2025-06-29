#include "config.hpp"
#ifdef CLI11_FOUND
#include <CLI/CLI.hpp>
#endif
#include <spdlog/spdlog.h>
#include <fstream>

namespace fast_td3 {

Config ConfigManager::parse_args(int argc, char* argv[]) {
    Config config;
    
#ifdef CLI11_FOUND
    CLI::App app{"FastTD3 - C++ Implementation"};
    
    // Environment settings
    app.add_option("--env_name", config.env_name, "Environment name");
    app.add_option("--agent", config.agent, "Agent type (fasttd3, fasttd3_simbav2)");
    app.add_option("--seed", config.seed, "Random seed");
    app.add_flag("--torch_deterministic", config.torch_deterministic, "Use deterministic PyTorch");
    app.add_flag("--cuda", config.cuda, "Use CUDA");
    app.add_option("--device_rank", config.device_rank, "Device rank");
    app.add_option("--exp_name", config.exp_name, "Experiment name");
    app.add_option("--project", config.project, "Project name");
    app.add_flag("--use_wandb", config.use_wandb, "Use Weights & Biases");
    app.add_option("--checkpoint_path", config.checkpoint_path, "Checkpoint path");
    
    // Training settings
    app.add_option("--num_envs", config.num_envs, "Number of environments");
    app.add_option("--num_eval_envs", config.num_eval_envs, "Number of evaluation environments");
    app.add_option("--total_timesteps", config.total_timesteps, "Total timesteps");
    app.add_option("--critic_learning_rate", config.critic_learning_rate, "Critic learning rate");
    app.add_option("--actor_learning_rate", config.actor_learning_rate, "Actor learning rate");
    app.add_option("--critic_learning_rate_end", config.critic_learning_rate_end, "Critic learning rate end");
    app.add_option("--actor_learning_rate_end", config.actor_learning_rate_end, "Actor learning rate end");
    app.add_option("--buffer_size", config.buffer_size, "Replay buffer size");
    app.add_option("--num_steps", config.num_steps, "Number of steps");
    app.add_option("--gamma", config.gamma, "Discount factor");
    app.add_option("--tau", config.tau, "Target smoothing coefficient");
    app.add_option("--batch_size", config.batch_size, "Batch size");
    app.add_option("--policy_noise", config.policy_noise, "Policy noise");
    app.add_option("--std_min", config.std_min, "Minimum noise std");
    app.add_option("--std_max", config.std_max, "Maximum noise std");
    app.add_option("--learning_starts", config.learning_starts, "Learning start timestep");
    app.add_option("--policy_frequency", config.policy_frequency, "Policy update frequency");
    app.add_option("--noise_clip", config.noise_clip, "Noise clip");
    app.add_option("--num_updates", config.num_updates, "Number of updates per step");
    app.add_option("--init_scale", config.init_scale, "Initial weight scale");
    app.add_option("--num_atoms", config.num_atoms, "Number of atoms");
    app.add_option("--v_min", config.v_min, "Minimum value");
    app.add_option("--v_max", config.v_max, "Maximum value");
    app.add_option("--critic_hidden_dim", config.critic_hidden_dim, "Critic hidden dimension");
    app.add_option("--actor_hidden_dim", config.actor_hidden_dim, "Actor hidden dimension");
    app.add_option("--critic_num_blocks", config.critic_num_blocks, "Critic number of blocks");
    app.add_option("--actor_num_blocks", config.actor_num_blocks, "Actor number of blocks");
    app.add_flag("--use_cdq", config.use_cdq, "Use clipped double Q-learning");
    app.add_option("--measure_burnin", config.measure_burnin, "Burn-in iterations");
    app.add_option("--eval_interval", config.eval_interval, "Evaluation interval");
    app.add_option("--render_interval", config.render_interval, "Render interval");
    app.add_flag("--compile", config.compile, "Use torch.compile");
    app.add_flag("--obs_normalization", config.obs_normalization, "Enable observation normalization");
    app.add_flag("--reward_normalization", config.reward_normalization, "Enable reward normalization");
    app.add_option("--max_grad_norm", config.max_grad_norm, "Maximum gradient norm");
    app.add_flag("--amp", config.amp, "Use automatic mixed precision");
    app.add_option("--amp_dtype", config.amp_dtype, "AMP data type");
    app.add_flag("--disable_bootstrap", config.disable_bootstrap, "Disable bootstrap");
    
    // Environment-specific settings
    app.add_flag("--use_domain_randomization", config.use_domain_randomization, "Use domain randomization");
    app.add_flag("--use_push_randomization", config.use_push_randomization, "Use push randomization");
    app.add_flag("--use_tuned_reward", config.use_tuned_reward, "Use tuned reward");
    app.add_option("--action_bounds", config.action_bounds, "Action bounds");
    app.add_option("--weight_decay", config.weight_decay, "Weight decay");
    app.add_option("--save_interval", config.save_interval, "Save interval");
    
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        spdlog::error("Error parsing arguments: {}", e.what());
        std::exit(app.exit(e));
    }
#else
    // Fallback: simple argument parsing without CLI11
    spdlog::info("CLI11 not available, using default configuration");
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            spdlog::info("FastTD3 - C++ Implementation");
            spdlog::info("Available options (CLI11 not available, using defaults):");
            spdlog::info("  --seed <int>           Random seed (default: {})", config.seed);
            spdlog::info("  --max-steps <int>      Maximum training steps (default: {})", config.total_timesteps);
            spdlog::info("  --batch-size <int>     Batch size (default: {})", config.batch_size);
            spdlog::info("  --env-name <string>    Environment name (default: {})", config.env_name);
            spdlog::info("  --log-level <string>   Log level (default: info)");
            std::exit(0);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            config.total_timesteps = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--env-name" && i + 1 < argc) {
            config.env_name = argv[++i];
        } else if (arg == "--log-level" && i + 1 < argc) {
            std::string level = argv[++i];
            if (level == "debug") spdlog::set_level(spdlog::level::debug);
            else if (level == "info") spdlog::set_level(spdlog::level::info);
            else if (level == "warn") spdlog::set_level(spdlog::level::warn);
            else if (level == "error") spdlog::set_level(spdlog::level::err);
        }
    }
#endif
    
    // Apply environment-specific configurations
    if (config.env_name.find("h1hand-") == 0 || config.env_name.find("h1-") == 0) {
        config = get_humanoid_bench_config(config.env_name);
    } else if (config.env_name.find("Isaac-") == 0) {
        config = get_isaaclab_config(config.env_name);
    } else {
        config = get_mujoco_playground_config(config.env_name);
    }
    
    return config;
}

Config ConfigManager::load_from_json(const std::string& filename) {
    Config config;
    std::ifstream file(filename);
    if (!file.is_open()) {
        spdlog::error("Could not open config file: {}", filename);
        return config;
    }
    
    nlohmann::json j;
    file >> j;
    
    // Load configuration from JSON
    if (j.contains("env_name")) config.env_name = j["env_name"];
    if (j.contains("agent")) config.agent = j["agent"];
    if (j.contains("seed")) config.seed = j["seed"];
    // ... add more fields as needed
    
    return config;
}

void ConfigManager::save_to_json(const Config& config, const std::string& filename) {
    nlohmann::json j;
    
    j["env_name"] = config.env_name;
    j["agent"] = config.agent;
    j["seed"] = config.seed;
    j["torch_deterministic"] = config.torch_deterministic;
    j["cuda"] = config.cuda;
    j["device_rank"] = config.device_rank;
    j["exp_name"] = config.exp_name;
    j["project"] = config.project;
    j["use_wandb"] = config.use_wandb;
    j["checkpoint_path"] = config.checkpoint_path;
    
    // Training settings
    j["num_envs"] = config.num_envs;
    j["num_eval_envs"] = config.num_eval_envs;
    j["total_timesteps"] = config.total_timesteps;
    j["critic_learning_rate"] = config.critic_learning_rate;
    j["actor_learning_rate"] = config.actor_learning_rate;
    j["buffer_size"] = config.buffer_size;
    j["gamma"] = config.gamma;
    j["batch_size"] = config.batch_size;
    // ... add more fields as needed
    
    std::ofstream file(filename);
    file << j.dump(4);
}

Config ConfigManager::get_default_config(const std::string& env_name) {
    Config config;
    config.env_name = env_name;
    return config;
}

Config ConfigManager::get_humanoid_bench_config(const std::string& env_name) {
    Config config = get_default_config(env_name);
    
    // HumanoidBench specific defaults
    config.total_timesteps = 100000;
    
    // Task-specific configurations
    if (env_name == "h1hand-reach-v0") {
        config.total_timesteps = 100000;
    } else if (env_name == "h1hand-balance-simple-v0") {
        config.total_timesteps = 200000;
    } else if (env_name == "h1hand-balance-hard-v0") {
        config.total_timesteps = 1000000;
    } else if (env_name == "h1hand-pole-v0") {
        config.total_timesteps = 150000;
    } else if (env_name == "h1hand-truck-v0") {
        config.total_timesteps = 500000;
    } else if (env_name == "h1hand-maze-v0") {
        config.v_min = -1000.0f;
        config.v_max = 1000.0f;
        config.total_timesteps = 1000000;
    } else if (env_name == "h1hand-push-v0") {
        config.v_min = -1000.0f;
        config.v_max = 1000.0f;
        config.total_timesteps = 1000000;
    } else if (env_name == "h1hand-basketball-v0") {
        config.v_min = -2000.0f;
        config.v_max = 2000.0f;
        config.total_timesteps = 250000;
    }
    
    return config;
}

Config ConfigManager::get_mujoco_playground_config(const std::string& env_name) {
    Config config = get_default_config(env_name);
    
    // MuJoCo Playground specific defaults
    config.v_min = -10.0f;
    config.v_max = 10.0f;
    config.buffer_size = 1024 * 10;
    config.num_envs = 1024;
    config.num_eval_envs = 1024;
    config.gamma = 0.97f;
    
    // Task-specific configurations
    if (env_name == "G1JoystickFlatTerrain") {
        config.total_timesteps = 100000;
    } else if (env_name == "T1JoystickFlatTerrain") {
        config.total_timesteps = 100000;
    } else if (env_name == "LeapCubeReorient") {
        config.num_steps = 3;
        config.gamma = 0.99f;
        config.policy_noise = 0.2f;
        config.v_min = -50.0f;
        config.v_max = 50.0f;
        config.use_cdq = false;
    } else if (env_name == "LeapCubeRotateZAxis") {
        config.num_steps = 1;
        config.policy_noise = 0.2f;
        config.gamma = 0.99f;
        config.v_min = -10.0f;
        config.v_max = 10.0f;
        config.use_cdq = false;
    }
    
    return config;
}

Config ConfigManager::get_isaaclab_config(const std::string& env_name) {
    Config config = get_default_config(env_name);
    
    // IsaacLab specific defaults
    config.v_min = -10.0f;
    config.v_max = 10.0f;
    config.buffer_size = 1024 * 10;
    config.num_envs = 4096;
    config.num_eval_envs = 4096;
    config.action_bounds = 1.0f;
    config.std_max = 0.4f;
    config.num_atoms = 251;
    config.render_interval = 0;  // IsaacLab does not support rendering
    config.total_timesteps = 100000;
    
    // Task-specific configurations
    if (env_name == "Isaac-Lift-Cube-Franka-v0") {
        config.num_updates = 8;
        config.v_min = -50.0f;
        config.v_max = 50.0f;
        config.std_max = 0.8f;
        config.num_envs = 1024;
        config.num_eval_envs = 1024;
        config.action_bounds = 3.0f;
        config.disable_bootstrap = true;
        config.total_timesteps = 20000;
    } else if (env_name == "Isaac-Velocity-Flat-G1-v0") {
        config.num_steps = 8;
        config.num_updates = 4;
        config.total_timesteps = 50000;
    }
    
    return config;
}

} // namespace fast_td3 