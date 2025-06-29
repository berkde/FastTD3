#pragma once

#include <vector>
#include <random>
#include <memory>

struct Experience {
    std::vector<float> state;
    std::vector<float> action;
    std::vector<float> next_state;
    float reward;
    bool done;
};

struct Batch {
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> actions;
    std::vector<std::vector<float>> next_states;
    std::vector<float> rewards;
    std::vector<bool> dones;
    int size;
};

class ReplayBuffer {
private:
    std::vector<Experience> buffer;
    size_t capacity;
    size_t current_size;
    size_t position;
    int state_dim;
    int action_dim;
    std::mt19937 rng;

public:
    ReplayBuffer(size_t capacity, int state_dim, int action_dim);
    
    void add(const float* state, const float* action, const float* next_state, 
             float reward, bool done);
    
    Batch sample(int batch_size);
    size_t size() const { return current_size; }
    bool is_full() const { return current_size == capacity; }
}; 