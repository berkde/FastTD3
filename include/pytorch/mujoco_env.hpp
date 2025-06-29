#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

namespace fast_td3 {

class MuJoCoEnvironment {
public:
    MuJoCoEnvironment(const std::string& env_name, int num_envs, torch::Device device);
    ~MuJoCoEnvironment();
    
    // Environment interface
    void reset();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions);
    
    // Getters
    torch::Tensor get_observations() const;
    int get_obs_dim() const;
    int get_act_dim() const;
    int get_num_envs() const;
    
    // Environment info
    std::string get_env_name() const;
    bool is_continuous() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Factory function for creating environments
std::unique_ptr<MuJoCoEnvironment> create_mujoco_env(
    const std::string& env_name, 
    int num_envs, 
    torch::Device device
);

// Available environment names
namespace envs {
    // Humanoid environments (newer versions)
    const std::string HUMANOID_V5 = "Humanoid-v5";
    const std::string HUMANOID_STANDUP_V5 = "HumanoidStandup-v5";
    const std::string WALKER_2D_V5 = "Walker2d-v5";
    const std::string HALF_CHEETAH_V5 = "HalfCheetah-v5";
    const std::string ANT_V5 = "Ant-v5";
    const std::string HOPPER_V5 = "Hopper-v5";
    
    // Legacy v2 environments (fallback)
    const std::string HUMANOID_V2 = "Humanoid-v2";
    const std::string HUMANOID_STANDUP_V2 = "HumanoidStandup-v2";
    const std::string WALKER_2D_V2 = "Walker2d-v2";
    const std::string HALF_CHEETAH_V2 = "HalfCheetah-v2";
    const std::string ANT_V2 = "Ant-v2";
    const std::string HOPPER_V2 = "Hopper-v2";
    
    // Custom humanoid environments (if available)
    const std::string H1HAND_STAND = "h1hand-stand-v0";
    const std::string H1HAND_REACH = "h1hand-reach-v0";
    const std::string H1HAND_BALANCE_SIMPLE = "h1hand-balance-simple-v0";
    const std::string H1HAND_BALANCE_HARD = "h1hand-balance-hard-v0";
}

} // namespace fast_td3 