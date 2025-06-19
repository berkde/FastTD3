#include "networks.hpp"
#include <torch/nn/init.h>
#include <spdlog/spdlog.h>

namespace fast_td3 {

DistributionalQNetwork::DistributionalQNetwork(
    int n_obs,
    int n_act,
    int num_atoms,
    float v_min,
    float v_max,
    int hidden_dim,
    torch::Device device
) : v_min(v_min), v_max(v_max), num_atoms(num_atoms) {
    
    net = torch::nn::Sequential(
        torch::nn::Linear(n_obs + n_act, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, hidden_dim / 2),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim / 2, hidden_dim / 4),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim / 4, num_atoms)
    );
    net->to(device);
    register_module("net", net);
}

torch::Tensor DistributionalQNetwork::forward(torch::Tensor obs, torch::Tensor actions) {
    auto x = torch::cat({obs, actions}, 1);
    return net->forward(x);
}

torch::Tensor DistributionalQNetwork::projection(
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor bootstrap,
    float discount,
    torch::Tensor q_support,
    torch::Device device
) {
    float delta_z = (v_max - v_min) / (num_atoms - 1);
    int batch_size = rewards.size(0);
    
    auto target_z = rewards.unsqueeze(1) + 
                   bootstrap.unsqueeze(1) * discount * q_support;
    target_z = torch::clamp(target_z, v_min, v_max);
    
    auto b = (target_z - v_min) / delta_z;
    auto l = torch::floor(b).to(torch::kLong);
    auto u = torch::ceil(b).to(torch::kLong);
    
    auto l_mask = (u > 0) & (l == u);
    auto u_mask = (l < (num_atoms - 1)) & (l == u);
    
    l = torch::where(l_mask, l - 1, l);
    u = torch::where(u_mask, u + 1, u);
    
    auto next_dist = torch::softmax(forward(obs, actions), 1);
    auto proj_dist = torch::zeros_like(next_dist);
    
    auto offset = torch::linspace(0, (batch_size - 1) * num_atoms, batch_size, device)
                     .unsqueeze(1)
                     .expand({batch_size, num_atoms})
                     .to(torch::kLong);
    
    proj_dist.view({-1}).index_add_(0, (l + offset).view({-1}), 
                                   (next_dist * (u.to(torch::kFloat) - b)).view({-1}));
    proj_dist.view({-1}).index_add_(0, (u + offset).view({-1}), 
                                   (next_dist * (b - l.to(torch::kFloat))).view({-1}));
    
    return proj_dist;
}

Critic::Critic(
    int n_obs,
    int n_act,
    int num_atoms,
    float v_min,
    float v_max,
    int hidden_dim,
    torch::Device device
) {
    qnet1 = std::make_shared<DistributionalQNetwork>(
        n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device);
    qnet2 = std::make_shared<DistributionalQNetwork>(
        n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device);
    
    q_support = torch::linspace(v_min, v_max, num_atoms, device);
    register_buffer("q_support", q_support);
}

std::pair<torch::Tensor, torch::Tensor> Critic::forward(torch::Tensor obs, torch::Tensor actions) {
    return {qnet1->forward(obs, actions), qnet2->forward(obs, actions)};
}

std::pair<torch::Tensor, torch::Tensor> Critic::projection(
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor bootstrap,
    float discount
) {
    auto q1_proj = qnet1->projection(obs, actions, rewards, bootstrap, discount, q_support, q_support.device());
    auto q2_proj = qnet2->projection(obs, actions, rewards, bootstrap, discount, q_support, q_support.device());
    
    return {q1_proj, q2_proj};
}

torch::Tensor Critic::get_value(torch::Tensor probs) {
    return torch::sum(probs * q_support, 1);
}

Actor::Actor(
    int n_obs,
    int n_act,
    int num_envs,
    float init_scale,
    int hidden_dim,
    float std_min,
    float std_max,
    torch::Device device
) : n_act(n_act), n_envs(num_envs) {
    
    net = torch::nn::Sequential(
        torch::nn::Linear(n_obs, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, hidden_dim / 2),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim / 2, hidden_dim / 4),
        torch::nn::ReLU()
    );
    net->to(device);
    
    fc_mu = torch::nn::Sequential(
        torch::nn::Linear(hidden_dim / 4, n_act),
        torch::nn::Tanh()
    );
    fc_mu->to(device);
    
    // Initialize weights
    torch::nn::init::normal_(fc_mu[0]->as<torch::nn::Linear>()->weight, 0.0, init_scale);
    torch::nn::init::constant_(fc_mu[0]->as<torch::nn::Linear>()->bias, 0.0);
    
    // Initialize noise scales
    noise_scales = torch::rand({num_envs, 1}, device) * (std_max - std_min) + std_min;
    this->std_min = torch::tensor(std_min, device);
    this->std_max = torch::tensor(std_max, device);
    
    register_module("net", net);
    register_module("fc_mu", fc_mu);
    register_buffer("noise_scales", noise_scales);
    register_buffer("std_min", this->std_min);
    register_buffer("std_max", this->std_max);
}

torch::Tensor Actor::forward(torch::Tensor obs) {
    auto x = net->forward(obs);
    return fc_mu->forward(x);
}

torch::Tensor Actor::explore(
    torch::Tensor obs,
    torch::Tensor dones,
    bool deterministic
) {
    // If dones is provided, resample noise for environments that are done
    if (dones.defined() && dones.sum().item<float>() > 0) {
        auto new_scales = torch::rand({n_envs, 1}, obs.device()) * 
                         (std_max - std_min) + std_min;
        
        auto dones_view = dones.view({-1, 1}) > 0;
        noise_scales = torch::where(dones_view, new_scales, noise_scales);
    }
    
    auto act = forward(obs);
    if (deterministic) {
        return act;
    }
    
    auto noise = torch::randn_like(act) * noise_scales;
    return act + noise;
}

} // namespace fast_td3 