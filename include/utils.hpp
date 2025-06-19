#pragma once

#include <torch/torch.h>
#include <string>
#include <memory>
#include "networks.hpp"
#include "normalizers.hpp"

namespace fast_td3 {

// Utility functions
torch::Tensor cpu_state(const torch::Tensor& tensor);
torch::Tensor clamp_tensor(torch::Tensor tensor, float min_val, float max_val);

// Parameter saving and loading
void save_params(
    int global_step,
    std::shared_ptr<Actor> actor,
    std::shared_ptr<Critic> critic,
    std::shared_ptr<Critic> critic_target,
    std::shared_ptr<EmpiricalNormalization> obs_normalizer,
    std::shared_ptr<EmpiricalNormalization> critic_obs_normalizer,
    const std::string& save_path
);

void load_params(
    std::shared_ptr<Actor> actor,
    std::shared_ptr<Critic> critic,
    std::shared_ptr<Critic> critic_target,
    std::shared_ptr<EmpiricalNormalization> obs_normalizer,
    std::shared_ptr<EmpiricalNormalization> critic_obs_normalizer,
    const std::string& load_path
);

// Training utilities
torch::Tensor compute_target_q(
    std::shared_ptr<Critic> critic_target,
    torch::Tensor next_obs,
    torch::Tensor next_actions,
    torch::Tensor rewards,
    torch::Tensor dones,
    float gamma,
    bool use_cdq = true
);

torch::Tensor compute_actor_loss(
    std::shared_ptr<Actor> actor,
    std::shared_ptr<Critic> critic,
    torch::Tensor obs,
    float policy_noise = 0.0f,
    float noise_clip = 0.5f
);

torch::Tensor compute_critic_loss(
    std::shared_ptr<Critic> critic,
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor target_q,
    bool use_cdq = true
);

// Environment utilities
torch::Device get_device(bool use_cuda, int device_rank = 0);
std::string get_device_string(torch::Device device);

// Logging utilities
void log_training_stats(
    int step,
    float actor_loss,
    float critic_loss,
    float total_reward,
    float episode_length,
    const std::string& log_file = ""
);

} // namespace fast_td3 