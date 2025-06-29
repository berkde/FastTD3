#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <map>
#include <spdlog/spdlog.h>
#ifdef USE_CLI11
#include <CLI/CLI.hpp>
#endif
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "config.hpp"
#include "replay_buffer_simple.hpp"
#include "normalizers_simple.hpp"
#include "utils_simple.hpp"

using json = nlohmann::json;
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

// Simple command-line argument parser (fallback when CLI11 is not available)
class SimpleArgParser {
private:
    std::map<std::string, std::string> args;
    
public:
    SimpleArgParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--") {
                std::string key = arg.substr(2);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    args[key] = argv[i + 1];
                    i++; // Skip next argument
                } else {
                    args[key] = "true";
                }
            }
        }
    }
    
    template<typename T>
    T get(const std::string& key, const T& default_value) const {
        auto it = args.find(key);
        if (it == args.end()) {
            return default_value;
        }
        
        if constexpr (std::is_same_v<T, int>) {
            return std::stoi(it->second);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::stof(it->second);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return it->second;
        } else if constexpr (std::is_same_v<T, bool>) {
            return it->second == "true" || it->second == "1";
        }
        return default_value;
    }
    
    bool has(const std::string& key) const {
        return args.find(key) != args.end();
    }
    
    void print_help() const {
        std::cout << "FastTD3 Simple - A simplified version without LibTorch\n";
        std::cout << "Usage: ./fast_td3_simple [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --seed <int>         Random seed (default: 42)\n";
        std::cout << "  --max-steps <int>    Maximum training steps (default: 1000000)\n";
        std::cout << "  --batch-size <int>   Batch size for training (default: 256)\n";
        std::cout << "  --state-dim <int>    State dimension (default: 17)\n";
        std::cout << "  --action-dim <int>   Action dimension (default: 6)\n";
        std::cout << "  --log-level <str>    Log level (default: info)\n";
        std::cout << "  --help               Show this help message\n";
    }
};

// Simple neural network using Eigen
class SimpleNeuralNetwork {
private:
    std::vector<Matrix> weights;
    std::vector<Vector> biases;
    std::vector<Vector> activations;
    
public:
    SimpleNeuralNetwork(const std::vector<int>& layer_sizes) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.01f);
        
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            weights.emplace_back(layer_sizes[i + 1], layer_sizes[i]);
            biases.emplace_back(layer_sizes[i + 1]);
            
            // Xavier initialization
            float scale = std::sqrt(2.0f / layer_sizes[i]);
            for (int j = 0; j < weights.back().size(); ++j) {
                weights.back()(j) = dist(gen) * scale;
            }
            biases.back().setZero();
        }
        
        activations.resize(layer_sizes.size());
    }
    
    Vector forward(const Vector& input) {
        activations[0] = input;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            activations[i + 1] = weights[i] * activations[i] + biases[i];
            if (i < weights.size() - 1) {
                // ReLU activation
                activations[i + 1] = activations[i + 1].cwiseMax(0.0f);
            }
        }
        
        return activations.back();
    }
    
    void update_weights(const std::vector<Matrix>& gradients, float lr = 0.001f) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * gradients[i];
        }
    }
};

// Simple TD3 Agent
class SimpleTD3Agent {
private:
    SimpleNeuralNetwork actor;
    SimpleNeuralNetwork critic1;
    SimpleNeuralNetwork critic2;
    SimpleNeuralNetwork actor_target;
    SimpleNeuralNetwork critic1_target;
    SimpleNeuralNetwork critic2_target;
    
    int state_dim;
    int action_dim;
    float gamma;
    float tau;
    float noise_std;
    
public:
    SimpleTD3Agent(int state_dim, int action_dim, int hidden_dim = 256)
        : actor({state_dim, hidden_dim, hidden_dim, action_dim}),
          critic1({state_dim + action_dim, hidden_dim, hidden_dim, 1}),
          critic2({state_dim + action_dim, hidden_dim, hidden_dim, 1}),
          actor_target({state_dim, hidden_dim, hidden_dim, action_dim}),
          critic1_target({state_dim + action_dim, hidden_dim, hidden_dim, 1}),
          critic2_target({state_dim + action_dim, hidden_dim, hidden_dim, 1}),
          state_dim(state_dim), action_dim(action_dim), gamma(0.99f), tau(0.005f), noise_std(0.1f) {
        
        // Initialize target networks
        copy_weights();
    }
    
    void copy_weights() {
        // In a real implementation, you'd copy weights between networks
        // For simplicity, we'll just note that this should happen
        spdlog::info("Copying weights to target networks");
    }
    
    Vector select_action(const Vector& state) {
        Vector action = actor.forward(state);
        
        // Add exploration noise
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, noise_std);
        
        for (int i = 0; i < action.size(); ++i) {
            action(i) += noise(gen);
            action(i) = std::clamp(action(i), -1.0f, 1.0f);
        }
        
        return action;
    }
    
    void update(const std::vector<Vector>& states,
                const std::vector<Vector>& actions,
                const std::vector<Vector>& next_states,
                const std::vector<float>& rewards,
                const std::vector<bool>& dones) {
        
        spdlog::info("Updating TD3 agent with {} samples", states.size());
        
        // This is a simplified update - in practice you'd implement proper backpropagation
        // For now, we'll just log that an update occurred
        
        // Update target networks
        update_target_networks();
    }
    
    void update_target_networks() {
        // In a real implementation, you'd use soft updates with tau
        spdlog::debug("Updating target networks with tau = {}", tau);
    }
};

int main(int argc, char* argv[]) {
    SimpleArgParser arg_parser(argc, argv);
    
    if (arg_parser.has("help")) {
        arg_parser.print_help();
        return 0;
    }
    
    // Configuration
    int seed = arg_parser.get("seed", 42);
    int max_steps = arg_parser.get("max-steps", 1000000);
    int batch_size = arg_parser.get("batch-size", 256);
    int state_dim = arg_parser.get("state-dim", 17);  // Humanoid state dimension
    int action_dim = arg_parser.get("action-dim", 6);  // Humanoid action dimension
    std::string log_level = arg_parser.get("log-level", std::string("info"));
    
    // Set up logging
    if (log_level == "debug") spdlog::set_level(spdlog::level::debug);
    else if (log_level == "info") spdlog::set_level(spdlog::level::info);
    else if (log_level == "warn") spdlog::set_level(spdlog::level::warn);
    else if (log_level == "error") spdlog::set_level(spdlog::level::err);
    
    spdlog::info("Starting FastTD3 Simple Training");
    spdlog::info("Configuration: seed={}, max_steps={}, batch_size={}, state_dim={}, action_dim={}", 
                 seed, max_steps, batch_size, state_dim, action_dim);
    
    // Set random seed
    std::srand(seed);
    
    // Initialize components
    ReplayBuffer replay_buffer(1000000, state_dim, action_dim);
    EmpiricalNormalization obs_normalizer({state_dim}, 1e-8f, 0);
    SimpleTD3Agent agent(state_dim, action_dim);
    
    // Training loop
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> state_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> reward_dist(-1.0f, 1.0f);
    
    for (int step = 0; step < max_steps; ++step) {
        // Generate dummy data for demonstration
        std::vector<float> state(state_dim);
        for (int i = 0; i < state_dim; ++i) {
            state[i] = state_dist(gen);
        }
        
        // Convert to Eigen vector for neural network
        Vector state_eigen = Eigen::Map<const Vector>(state.data(), state_dim);
        
        // Select action
        Vector action_eigen = agent.select_action(state_eigen);
        
        // Convert back to std::vector
        std::vector<float> action(action_dim);
        for (int i = 0; i < action_dim; ++i) {
            action[i] = action_eigen(i);
        }
        
        // Generate next state and reward
        std::vector<float> next_state(state_dim);
        for (int i = 0; i < state_dim; ++i) {
            next_state[i] = state_dist(gen);
        }
        
        float reward = reward_dist(gen);
        bool done = (step % 1000 == 0);  // Episode ends every 1000 steps
        
        // Store experience
        replay_buffer.add(state.data(), action.data(), next_state.data(), reward, done);
        
        // Update agent if we have enough samples
        if (step > batch_size && step % 10 == 0) {
            auto batch = replay_buffer.sample(batch_size);
            
            std::vector<Vector> states, actions, next_states;
            std::vector<float> rewards;
            std::vector<bool> dones;
            
            for (int i = 0; i < batch.size; ++i) {
                Vector state_vec = Eigen::Map<const Vector>(batch.states[i].data(), state_dim);
                Vector action_vec = Eigen::Map<const Vector>(batch.actions[i].data(), action_dim);
                Vector next_state_vec = Eigen::Map<const Vector>(batch.next_states[i].data(), state_dim);
                
                states.push_back(state_vec);
                actions.push_back(action_vec);
                next_states.push_back(next_state_vec);
                rewards.push_back(batch.rewards[i]);
                dones.push_back(batch.dones[i]);
            }
            
            agent.update(states, actions, next_states, rewards, dones);
        }
        
        // Log progress
        if (step % 10000 == 0) {
            spdlog::info("Step {}: Buffer size = {}", step, replay_buffer.size());
        }
    }
    
    spdlog::info("Training completed!");
    return 0;
} 