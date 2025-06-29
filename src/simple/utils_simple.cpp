#include "utils_simple.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>

namespace fast_td3 {

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
    return std::string(buffer);
}

void set_random_seed(int seed) {
    std::srand(seed);
    spdlog::info("Random seed set to: {}", seed);
}

float clip(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

std::vector<float> clip_actions(const std::vector<float>& actions, float min_val, float max_val) {
    std::vector<float> clipped(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        clipped[i] = clip(actions[i], min_val, max_val);
    }
    return clipped;
}

float compute_td_error(float current_q, float target_q) {
    return target_q - current_q;
}

float huber_loss(float error, float delta) {
    if (std::abs(error) <= delta) {
        return 0.5f * error * error;
    } else {
        return delta * (std::abs(error) - 0.5f * delta);
    }
}

std::vector<float> soft_update(const std::vector<float>& target, 
                              const std::vector<float>& source, 
                              float tau) {
    std::vector<float> result(target.size());
    for (size_t i = 0; i < target.size(); ++i) {
        result[i] = tau * source[i] + (1.0f - tau) * target[i];
    }
    return result;
}

} // namespace fast_td3 