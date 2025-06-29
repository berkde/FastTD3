#include "normalizers_simple.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

EmpiricalNormalization::EmpiricalNormalization(std::vector<int64_t> shape, float eps, int until)
    : eps(eps), until(until), update_count(0) {
    
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    
    mean.resize(size, 0.0f);
    std.resize(size, 1.0f);
    count.resize(size, 0);
}

void EmpiricalNormalization::update(const std::vector<float>& batch) {
    if (update_count >= until) {
        return;
    }
    
    int64_t batch_size = batch.size() / mean.size();
    if (batch_size == 0) return;
    
    // Compute batch mean and std
    std::vector<float> batch_mean(mean.size(), 0.0f);
    std::vector<float> batch_std(mean.size(), 0.0f);
    
    for (size_t i = 0; i < mean.size(); ++i) {
        // Compute mean
        float sum = 0.0f;
        for (int j = 0; j < batch_size; ++j) {
            sum += batch[j * mean.size() + i];
        }
        batch_mean[i] = sum / batch_size;
        
        // Compute std
        float var_sum = 0.0f;
        for (int j = 0; j < batch_size; ++j) {
            float diff = batch[j * mean.size() + i] - batch_mean[i];
            var_sum += diff * diff;
        }
        batch_std[i] = std::sqrt(var_sum / batch_size + eps);
    }
    
    // Update running statistics
    for (size_t i = 0; i < mean.size(); ++i) {
        if (count[i] == 0) {
            mean[i] = batch_mean[i];
            std[i] = batch_std[i];
            count[i] = batch_size;
        } else {
            float old_mean = mean[i];
            float old_std = std[i];
            int64_t old_count = count[i];
            
            // Update mean
            mean[i] = (old_mean * old_count + batch_mean[i] * batch_size) / (old_count + batch_size);
            
            // Update std (simplified)
            std[i] = std::sqrt((old_std * old_std * old_count + batch_std[i] * batch_std[i] * batch_size) / (old_count + batch_size));
            
            count[i] = old_count + batch_size;
        }
    }
    
    update_count++;
}

std::vector<float> EmpiricalNormalization::normalize(const std::vector<float>& x) {
    std::vector<float> normalized(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        size_t feature_idx = i % mean.size();
        normalized[i] = (x[i] - mean[feature_idx]) / (std[feature_idx] + eps);
    }
    
    return normalized;
}

void EmpiricalNormalization::reset() {
    std::fill(mean.begin(), mean.end(), 0.0f);
    std::fill(std.begin(), std.end(), 1.0f);
    std::fill(count.begin(), count.end(), 0);
    update_count = 0;
} 