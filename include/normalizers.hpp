#pragma once

#include <torch/torch.h>
#include <memory>

namespace fast_td3 {

class EmpiricalNormalization : public torch::nn::Module {
public:
    EmpiricalNormalization(
        std::vector<int64_t> shape,
        torch::Device device,
        float eps = 1e-2f,
        int until = -1
    );

    torch::Tensor forward(torch::Tensor x, bool center = true);
    void update(torch::Tensor x);
    torch::Tensor inverse(torch::Tensor y);

    torch::Tensor get_mean() const { return mean; }
    torch::Tensor get_std() const { return std; }

private:
    torch::Tensor mean;
    torch::Tensor std;
    torch::Tensor count;
    float eps;
    int until;
    int update_count;
    torch::Device device;
};

class RewardNormalizer : public torch::nn::Module {
public:
    RewardNormalizer(
        float gamma,
        torch::Device device,
        float g_max = 10.0f,
        float epsilon = 1e-8f
    );

    torch::Tensor forward(torch::Tensor rewards);
    void update_stats(torch::Tensor rewards, torch::Tensor dones);
    torch::Tensor scale_reward(torch::Tensor rewards);

private:
    float gamma;
    torch::Device device;
    float g_max;
    float epsilon;
    torch::Tensor returns;
    torch::Tensor ret_rms;
};

} // namespace fast_td3 