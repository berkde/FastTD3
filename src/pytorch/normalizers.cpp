#include "normalizers.hpp"
#include <spdlog/spdlog.h>

namespace fast_td3 {

EmpiricalNormalization::EmpiricalNormalization(
    std::vector<int64_t> shape,
    torch::Device device,
    float eps,
    int until
) : eps(eps), until(until), update_count(0), device(device) {
    
    mean = torch::zeros(shape, device);
    std = torch::ones(shape, device);
    count = torch::zeros({1}, device);
    
    register_buffer("mean", mean);
    register_buffer("std", std);
    register_buffer("count", count);
}

torch::Tensor EmpiricalNormalization::forward(torch::Tensor x, bool center) {
    if (center) {
        return (x - mean) / (std + eps);
    } else {
        return x / (std + eps);
    }
}

void EmpiricalNormalization::update(torch::Tensor x) {
    if (until >= 0 && update_count >= until) {
        return;
    }
    
    auto batch_mean = torch::mean(x, 0);
    auto batch_std = torch::std(x, 0);
    auto batch_count = x.size(0);
    
    if (count.item<float>() == 0) {
        mean = batch_mean;
        std = batch_std;
        count = torch::tensor(batch_count, this->device);
    } else {
        auto delta = batch_mean - mean;
        auto tot_count = count + batch_count;
        
        auto new_mean = mean + delta * batch_count / tot_count;
        auto m_a = std * std * count;
        auto m_b = batch_std * batch_std * batch_count;
        auto M2 = m_a + m_b + delta * delta * count * batch_count / tot_count;
        auto new_std = torch::sqrt(M2 / tot_count);
        
        mean = new_mean;
        std = new_std;
        count = tot_count;
    }
    
    update_count++;
}

torch::Tensor EmpiricalNormalization::inverse(torch::Tensor y) {
    return y * (std + eps) + mean;
}

RewardNormalizer::RewardNormalizer(
    float gamma,
    torch::Device device,
    float g_max,
    float epsilon
) : gamma(gamma), device(device), g_max(g_max), epsilon(epsilon) {
    
    returns = torch::zeros({1}, device);
    ret_rms = torch::ones({1}, device);
    
    register_buffer("returns", returns);
    register_buffer("ret_rms", ret_rms);
}

torch::Tensor RewardNormalizer::forward(torch::Tensor rewards) {
    return scale_reward(rewards);
}

void RewardNormalizer::update_stats(torch::Tensor rewards, torch::Tensor dones) {
    auto ret = returns.clone();
    auto ret_rms_val = ret_rms.clone();
    
    for (int i = 0; i < rewards.size(0); i++) {
        ret = ret * gamma + rewards[i];
        ret_rms_val = torch::sqrt(ret_rms_val * ret_rms_val * 0.99f + ret * ret * 0.01f);
        
        if (dones[i].item<bool>()) {
            ret = torch::zeros_like(ret);
        }
    }
    
    returns = ret;
    ret_rms = ret_rms_val;
}

torch::Tensor RewardNormalizer::scale_reward(torch::Tensor rewards) {
    return torch::clamp(rewards / (ret_rms + epsilon), -g_max, g_max);
}

} // namespace fast_td3 