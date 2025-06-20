#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "replay_buffer_simple.hpp"
#include "normalizers_simple.hpp"
#include "utils_simple.hpp"

using json = nlohmann::json;
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

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
    CLI::App app{"FastTD3 Simple - A simplified version without LibTorch"};
    
    // Configuration
    int seed = 42;
    int max_steps = 1000000;
    int batch_size = 256;
    int state_dim = 17;  // Humanoid state dimension
    int action_dim = 6;  // Humanoid action dimension
    std::string log_level = "info";
    
    app.add_option("--seed", seed, "Random seed");
    app.add_option("--max-steps", max_steps, "Maximum training steps");
    app.add_option("--batch-size", batch_size, "Batch size for training");
    app.add_option("--state-dim", state_dim, "State dimension");
    app.add_option("--action-dim", action_dim, "Action dimension");
    app.add_option("--log-level", log_level, "Log level (debug, info, warn, error)");
    
    CLI11_PARSE(app, argc, argv);
    
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
        
        // Store in replay buffer
        replay_buffer.add(state.data(), action.data(), next_state.data(), reward, done);
        
        // Update if we have enough samples
        if (step > 1000 && step % 4 == 0) {
            auto batch = replay_buffer.sample(batch_size);
            
            std::vector<Vector> states, actions, next_states;
            std::vector<float> rewards;
            std::vector<bool> dones;
            
            for (int i = 0; i < batch.size; ++i) {
                states.emplace_back(Eigen::Map<const Vector>(batch.states[i].data(), state_dim));
                actions.emplace_back(Eigen::Map<const Vector>(batch.actions[i].data(), action_dim));
                next_states.emplace_back(Eigen::Map<const Vector>(batch.next_states[i].data(), state_dim));
                rewards.push_back(batch.rewards[i]);
                dones.push_back(batch.dones[i]);
            }
            
            agent.update(states, actions, next_states, rewards, dones);
        }
        
        if (step % 10000 == 0) {
            spdlog::info("Step {}: Replay buffer size = {}", step, replay_buffer.size());
        }
    }
    
    spdlog::info("Training completed!");
    return 0;
} 