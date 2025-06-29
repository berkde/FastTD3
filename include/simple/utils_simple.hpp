#pragma once

#include <string>
#include <vector>

namespace fast_td3 {

// Utility functions
std::string get_timestamp();
void set_random_seed(int seed);
float clip(float value, float min_val, float max_val);
std::vector<float> clip_actions(const std::vector<float>& actions, float min_val, float max_val);
float compute_td_error(float current_q, float target_q);
float huber_loss(float error, float delta);
std::vector<float> soft_update(const std::vector<float>& target, 
                              const std::vector<float>& source, 
                              float tau);

} // namespace fast_td3 