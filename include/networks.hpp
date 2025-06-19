#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>

namespace fast_td3 {

class DistributionalQNetwork : public torch::nn::Module {
public:
    DistributionalQNetwork(
        int n_obs,
        int n_act,
        int num_atoms,
        float v_min,
        float v_max,
        int hidden_dim,
        torch::Device device = torch::kCPU
    );

    torch::Tensor forward(torch::Tensor obs, torch::Tensor actions);
    
    torch::Tensor projection(
        torch::Tensor obs,
        torch::Tensor actions,
        torch::Tensor rewards,
        torch::Tensor bootstrap,
        float discount,
        torch::Tensor q_support,
        torch::Device device
    );

private:
    torch::nn::Sequential net{nullptr};
    float v_min;
    float v_max;
    int num_atoms;
};

class Critic : public torch::nn::Module {
public:
    Critic(
        int n_obs,
        int n_act,
        int num_atoms,
        float v_min,
        float v_max,
        int hidden_dim,
        torch::Device device = torch::kCPU
    );

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs, torch::Tensor actions);
    
    std::pair<torch::Tensor, torch::Tensor> projection(
        torch::Tensor obs,
        torch::Tensor actions,
        torch::Tensor rewards,
        torch::Tensor bootstrap,
        float discount
    );

    torch::Tensor get_value(torch::Tensor probs);

private:
    std::shared_ptr<DistributionalQNetwork> qnet1;
    std::shared_ptr<DistributionalQNetwork> qnet2;
    torch::Tensor q_support;
};

class Actor : public torch::nn::Module {
public:
    Actor(
        int n_obs,
        int n_act,
        int num_envs,
        float init_scale,
        int hidden_dim,
        float std_min = 0.05f,
        float std_max = 0.8f,
        torch::Device device = torch::kCPU
    );

    torch::Tensor forward(torch::Tensor obs);
    
    torch::Tensor explore(
        torch::Tensor obs,
        torch::Tensor dones = torch::Tensor(),
        bool deterministic = false
    );

private:
    torch::nn::Sequential net{nullptr};
    torch::nn::Sequential fc_mu{nullptr};
    torch::Tensor noise_scales;
    torch::Tensor std_min;
    torch::Tensor std_max;
    int n_act;
    int n_envs;
};

} // namespace fast_td3 