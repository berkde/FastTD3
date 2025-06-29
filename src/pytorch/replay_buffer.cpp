#include "replay_buffer.hpp"
#include <spdlog/spdlog.h>

namespace fast_td3 {

SimpleReplayBuffer::SimpleReplayBuffer(
    int n_env,
    int buffer_size,
    int n_obs,
    int n_act,
    int n_critic_obs,
    bool asymmetric_obs,
    bool playground_mode,
    int n_steps,
    float gamma,
    torch::Device device
) : n_env(n_env), buffer_size(buffer_size), n_obs(n_obs), n_act(n_act), 
    n_critic_obs(n_critic_obs), asymmetric_obs(asymmetric_obs), 
    playground_mode(playground_mode && asymmetric_obs), gamma(gamma), 
    n_steps(n_steps), device(device), ptr(0) {
    
    observations = torch::zeros({n_env, buffer_size, n_obs}, device);
    actions = torch::zeros({n_env, buffer_size, n_act}, device);
    rewards = torch::zeros({n_env, buffer_size}, device);
    dones = torch::zeros({n_env, buffer_size}, device);
    truncations = torch::zeros({n_env, buffer_size}, device);
    next_observations = torch::zeros({n_env, buffer_size, n_obs}, device);
    
    if (asymmetric_obs) {
        if (playground_mode) {
            privileged_obs_size = n_critic_obs - n_obs;
            privileged_observations = torch::zeros({n_env, buffer_size, privileged_obs_size}, device);
            next_privileged_observations = torch::zeros({n_env, buffer_size, privileged_obs_size}, device);
        } else {
            critic_observations = torch::zeros({n_env, buffer_size, n_critic_obs}, device);
            next_critic_observations = torch::zeros({n_env, buffer_size, n_critic_obs}, device);
        }
    }
}

void SimpleReplayBuffer::extend(const Transition& transition) {
    int current_ptr = ptr % buffer_size;
    
    observations.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                           transition.observations.detach());
    actions.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                      transition.actions.detach());
    rewards.index_put_({torch::indexing::Slice(), current_ptr}, transition.rewards.detach());
    dones.index_put_({torch::indexing::Slice(), current_ptr}, transition.dones.detach());
    truncations.index_put_({torch::indexing::Slice(), current_ptr}, transition.truncations.detach());
    next_observations.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                                transition.next_observations.detach());
    
    if (asymmetric_obs) {
        if (playground_mode) {
            auto privileged_obs = transition.critic_observations.index({torch::indexing::Slice(), 
                                                                      torch::indexing::Slice(n_obs, torch::indexing::None)}).detach();
            auto next_privileged_obs = transition.next_critic_observations.index({torch::indexing::Slice(), 
                                                                                torch::indexing::Slice(n_obs, torch::indexing::None)}).detach();
            
            privileged_observations.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                                             privileged_obs);
            next_privileged_observations.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                                                  next_privileged_obs);
        } else {
            critic_observations.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                                         transition.critic_observations.detach());
            next_critic_observations.index_put_({torch::indexing::Slice(), current_ptr, torch::indexing::Slice()}, 
                                              transition.next_critic_observations.detach());
        }
    }
    
    ptr++;
}

Transition SimpleReplayBuffer::sample(int batch_size) {
    Transition transition;
    
    if (n_steps == 1) {
        auto indices = torch::randint(0, std::min(buffer_size, ptr), 
                                    {n_env, batch_size}, 
                                    torch::TensorOptions().dtype(torch::kInt64).device(device));
        
        auto obs_indices = indices.unsqueeze(-1).expand({-1, -1, n_obs});
        auto act_indices = indices.unsqueeze(-1).expand({-1, -1, n_act});
        
        transition.observations = torch::gather(observations, 1, obs_indices)
                                    .reshape({n_env * batch_size, n_obs});
        transition.next_observations = torch::gather(next_observations, 1, obs_indices)
                                         .reshape({n_env * batch_size, n_obs});
        transition.actions = torch::gather(actions, 1, act_indices)
                               .reshape({n_env * batch_size, n_act});
        transition.rewards = torch::gather(rewards, 1, indices)
                               .reshape({n_env * batch_size});
        transition.dones = torch::gather(dones, 1, indices)
                             .reshape({n_env * batch_size});
        transition.truncations = torch::gather(truncations, 1, indices)
                                   .reshape({n_env * batch_size});
        
        if (asymmetric_obs) {
            if (playground_mode) {
                auto priv_obs_indices = indices.unsqueeze(-1).expand({-1, -1, privileged_obs_size});
                auto privileged_obs = torch::gather(privileged_observations, 1, priv_obs_indices)
                                        .reshape({n_env * batch_size, privileged_obs_size});
                auto next_privileged_obs = torch::gather(next_privileged_observations, 1, priv_obs_indices)
                                             .reshape({n_env * batch_size, privileged_obs_size});
                
                transition.critic_observations = torch::cat({transition.observations, privileged_obs}, 1);
                transition.next_critic_observations = torch::cat({transition.next_observations, next_privileged_obs}, 1);
            } else {
                auto critic_obs_indices = indices.unsqueeze(-1).expand({-1, -1, n_critic_obs});
                transition.critic_observations = torch::gather(critic_observations, 1, critic_obs_indices)
                                                   .reshape({n_env * batch_size, n_critic_obs});
                transition.next_critic_observations = torch::gather(next_critic_observations, 1, critic_obs_indices)
                                                        .reshape({n_env * batch_size, n_critic_obs});
            }
        } else {
            transition.critic_observations = transition.observations;
            transition.next_critic_observations = transition.next_observations;
        }
    } else {
        // Handle n-step returns (simplified implementation)
        // In a full implementation, this would compute n-step returns
        // For now, we'll use single-step sampling
        spdlog::warn("N-step sampling not fully implemented, using single-step");
        return sample(1);
    }
    
    return transition;
}

} // namespace fast_td3 