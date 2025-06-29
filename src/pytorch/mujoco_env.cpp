#include "mujoco_env.hpp"
#include <spdlog/spdlog.h>
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fast_td3 {

class MuJoCoEnvironment::Impl {
public:
    Impl(const std::string& env_name, int num_envs, torch::Device device) 
        : env_name(env_name), num_envs(num_envs), device(device) {
        
        // Initialize Python interpreter if not already done
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
        }
        
        try {
            // Import required modules
            py::module gym = py::module::import("gymnasium");
            py::module np = py::module::import("numpy");
            
            // Create environment - try newer version first, fallback to v2
            py::object env_class = gym.attr("make");
            bool env_created = false;
            
            // Try v5 first
            try {
                env = env_class(env_name, "v5");
                spdlog::info("Using MuJoCo environment v5: {}", env_name);
                env_created = true;
            } catch (const py::error_already_set& e) {
                // Clear the error by calling PyErr_Clear
                PyErr_Clear();
            }
            
            // If v5 failed, try v2
            if (!env_created) {
                try {
                    env = env_class(env_name, "v2");
                    spdlog::info("Using MuJoCo environment v2: {}", env_name);
                    env_created = true;
                } catch (const py::error_already_set& e) {
                    spdlog::error("Failed to create environment with both v5 and v2: {}", e.what());
                    throw std::runtime_error("Failed to create MuJoCo environment");
                }
            }
            
            // Get environment info
            py::object info = env.attr("spec");
            obs_space = info.attr("observation_space");
            act_space = info.attr("action_space");
            
            // Extract dimensions
            obs_dim = obs_space.attr("shape").attr("__getitem__")(0).cast<int>();
            act_dim = act_space.attr("shape").attr("__getitem__")(0).cast<int>();
            
            // Check if continuous action space
            is_continuous_env = act_space.attr("__class__").attr("__name__").cast<std::string>() == "Box";
            
            spdlog::info("MuJoCo environment created: {} (obs_dim={}, act_dim={}, continuous={})", 
                        env_name, obs_dim, act_dim, is_continuous_env);
            
            // Initialize observations
            reset();
            
        } catch (const py::error_already_set& e) {
            spdlog::error("Failed to create MuJoCo environment: {}", e.what());
            throw std::runtime_error("Failed to create MuJoCo environment");
        }
    }
    
    ~Impl() {
        // Cleanup will be handled by Python's garbage collector
    }
    
    void reset() {
        try {
            py::object result = env.attr("reset")();
            py::object obs = result.attr("__getitem__")(0);
            
            // Convert observation to tensor
            py::array_t<float> obs_array = obs.cast<py::array_t<float>>();
            auto obs_tensor = torch::from_blob(
                obs_array.mutable_data(), 
                {1, obs_dim}, 
                torch::kFloat32
            ).clone().to(device);
            
            // For multiple environments, we'll need to handle this differently
            // For now, we'll just use the single environment observation
            observations = obs_tensor;
            
        } catch (const py::error_already_set& e) {
            spdlog::error("Failed to reset environment: {}", e.what());
            throw std::runtime_error("Failed to reset environment");
        }
    }
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions) {
        try {
            // Convert tensor to numpy array
            auto actions_cpu = actions.cpu();
            py::array_t<float> actions_array(
                {actions_cpu.size(0), actions_cpu.size(1)},
                actions_cpu.data_ptr<float>()
            );
            
            // Take step in environment
            py::object result = env.attr("step")(actions_array);
            
            // Extract results
            py::object obs = result.attr("__getitem__")(0);
            py::object reward = result.attr("__getitem__")(1);
            py::object terminated = result.attr("__getitem__")(2);
            py::object truncated = result.attr("__getitem__")(3);
            
            // Convert to tensors
            py::array_t<float> obs_array = obs.cast<py::array_t<float>>();
            auto next_obs = torch::from_blob(
                obs_array.mutable_data(), 
                {1, obs_dim}, 
                torch::kFloat32
            ).clone().to(device);
            
            auto rewards = torch::tensor({reward.cast<float>()}, device);
            auto dones = torch::tensor({terminated.cast<bool>()}, device);
            auto truncations = torch::tensor({truncated.cast<bool>()}, device);
            
            // Update observations
            observations = next_obs;
            
            return {next_obs, rewards, dones, truncations};
            
        } catch (const py::error_already_set& e) {
            spdlog::error("Failed to step environment: {}", e.what());
            throw std::runtime_error("Failed to step environment");
        }
    }
    
    torch::Tensor get_observations() const {
        return observations;
    }
    
    int get_obs_dim() const { return obs_dim; }
    int get_act_dim() const { return act_dim; }
    int get_num_envs() const { return num_envs; }
    std::string get_env_name() const { return env_name; }
    bool is_continuous() const { return is_continuous_env; }
    
private:
    std::string env_name;
    int num_envs;
    torch::Device device;
    py::object env;
    py::object obs_space;
    py::object act_space;
    int obs_dim;
    int act_dim;
    bool is_continuous_env;
    torch::Tensor observations;
};

// MuJoCoEnvironment implementation
MuJoCoEnvironment::MuJoCoEnvironment(const std::string& env_name, int num_envs, torch::Device device)
    : pImpl(std::make_unique<Impl>(env_name, num_envs, device)) {}

MuJoCoEnvironment::~MuJoCoEnvironment() = default;

void MuJoCoEnvironment::reset() {
    pImpl->reset();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MuJoCoEnvironment::step(torch::Tensor actions) {
    return pImpl->step(actions);
}

torch::Tensor MuJoCoEnvironment::get_observations() const {
    return pImpl->get_observations();
}

int MuJoCoEnvironment::get_obs_dim() const {
    return pImpl->get_obs_dim();
}

int MuJoCoEnvironment::get_act_dim() const {
    return pImpl->get_act_dim();
}

int MuJoCoEnvironment::get_num_envs() const {
    return pImpl->get_num_envs();
}

std::string MuJoCoEnvironment::get_env_name() const {
    return pImpl->get_env_name();
}

bool MuJoCoEnvironment::is_continuous() const {
    return pImpl->is_continuous();
}

// Factory function
std::unique_ptr<MuJoCoEnvironment> create_mujoco_env(
    const std::string& env_name, 
    int num_envs, 
    torch::Device device
) {
    return std::make_unique<MuJoCoEnvironment>(env_name, num_envs, device);
}

} // namespace fast_td3 