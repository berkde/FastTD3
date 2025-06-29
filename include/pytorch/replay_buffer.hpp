#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace fast_td3 {

struct Transition {
    torch::Tensor observations;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor dones;
    torch::Tensor truncations;
    torch::Tensor next_observations;
    torch::Tensor critic_observations;
    torch::Tensor next_critic_observations;
};

class SimpleReplayBuffer : public torch::nn::Module {
public:
    SimpleReplayBuffer(
        int n_env,
        int buffer_size,
        int n_obs,
        int n_act,
        int n_critic_obs,
        bool asymmetric_obs = false,
        bool playground_mode = false,
        int n_steps = 1,
        float gamma = 0.99f,
        torch::Device device = torch::kCPU
    );

    void extend(const Transition& transition);
    Transition sample(int batch_size);

    int get_ptr() const { return ptr; }
    int get_buffer_size() const { return buffer_size; }

private:
    int n_env;
    int buffer_size;
    int n_obs;
    int n_act;
    int n_critic_obs;
    bool asymmetric_obs;
    bool playground_mode;
    float gamma;
    int n_steps;
    torch::Device device;

    torch::Tensor observations;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor dones;
    torch::Tensor truncations;
    torch::Tensor next_observations;
    
    // For asymmetric observations
    torch::Tensor privileged_observations;
    torch::Tensor next_privileged_observations;
    torch::Tensor critic_observations;
    torch::Tensor next_critic_observations;
    int privileged_obs_size;

    int ptr;
};

} // namespace fast_td3 