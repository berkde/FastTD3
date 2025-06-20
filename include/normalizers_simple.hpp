#pragma once

#include <vector>
#include <cmath>

class EmpiricalNormalization {
private:
    std::vector<float> mean;
    std::vector<float> std;
    std::vector<int64_t> count;
    float eps;
    int until;
    int update_count;

public:
    EmpiricalNormalization(std::vector<int64_t> shape, float eps, int until);
    
    void update(const std::vector<float>& batch);
    std::vector<float> normalize(const std::vector<float>& x);
    void reset();
    
    const std::vector<float>& get_mean() const { return mean; }
    const std::vector<float>& get_std() const { return std; }
}; 