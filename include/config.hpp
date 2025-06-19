#pragma once

#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace fast_td3 {

struct Config {
    // Environment settings
    std::string env_name = "h1hand-stand-v0";
    std::string agent = "fasttd3";
    int seed = 1;
    bool torch_deterministic = true;
    bool cuda = true;
    int device_rank = 0;
    std::string exp_name = "FastTD3";
    std::string project = "FastTD3";
    bool use_wandb = true;
    std::string checkpoint_path = "";
    
    // Training settings
    int num_envs = 128;
    int num_eval_envs = 128;
    int total_timesteps = 150000;
    float critic_learning_rate = 3e-4f;
    float actor_learning_rate = 3e-4f;
    float critic_learning_rate_end = 3e-4f;
    float actor_learning_rate_end = 3e-4f;
    int buffer_size = 1024 * 50;
    int num_steps = 1;
    float gamma = 0.99f;
    float tau = 0.1f;
    int batch_size = 32768;
    float policy_noise = 0.001f;
    float std_min = 0.001f;
    float std_max = 0.4f;
    int learning_starts = 10;
    int policy_frequency = 2;
    float noise_clip = 0.5f;
    int num_updates = 2;
    float init_scale = 0.01f;
    int num_atoms = 101;
    float v_min = -250.0f;
    float v_max = 250.0f;
    int critic_hidden_dim = 1024;
    int actor_hidden_dim = 512;
    int critic_num_blocks = 2;
    int actor_num_blocks = 1;
    bool use_cdq = true;
    int measure_burnin = 3;
    int eval_interval = 5000;
    int render_interval = 5000;
    bool compile = true;
    bool obs_normalization = true;
    bool reward_normalization = false;
    float max_grad_norm = 0.0f;
    bool amp = true;
    std::string amp_dtype = "bf16";
    bool disable_bootstrap = false;
    
    // Environment-specific settings
    bool use_domain_randomization = false;
    bool use_push_randomization = false;
    bool use_tuned_reward = false;
    float action_bounds = 1.0f;
    float weight_decay = 0.1f;
    int save_interval = 5000;
    
    // SimbaV2 specific settings
    float scaler_init = 0.0f;
    float scaler_scale = 0.0f;
    float alpha_init = 0.0f;
    float alpha_scale = 0.0f;
    int expansion = 4;
    float c_shift = 3.0f;
};

class ConfigManager {
public:
    static Config parse_args(int argc, char* argv[]);
    static Config load_from_json(const std::string& filename);
    static void save_to_json(const Config& config, const std::string& filename);
    
    static Config get_default_config(const std::string& env_name);
    static Config get_humanoid_bench_config(const std::string& env_name);
    static Config get_mujoco_playground_config(const std::string& env_name);
    static Config get_isaaclab_config(const std::string& env_name);
};

} // namespace fast_td3 