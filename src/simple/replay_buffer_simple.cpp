#include "replay_buffer_simple.hpp"
#include <algorithm>
#include <random>

ReplayBuffer::ReplayBuffer(size_t capacity, int state_dim, int action_dim)
    : capacity(capacity), current_size(0), position(0), 
      state_dim(state_dim), action_dim(action_dim) {
    buffer.reserve(capacity);
    rng.seed(std::random_device{}());
}

void ReplayBuffer::add(const float* state, const float* action, const float* next_state, 
                      float reward, bool done) {
    Experience exp;
    exp.state.assign(state, state + state_dim);
    exp.action.assign(action, action + action_dim);
    exp.next_state.assign(next_state, next_state + state_dim);
    exp.reward = reward;
    exp.done = done;
    
    if (current_size < capacity) {
        buffer.push_back(exp);
        current_size++;
    } else {
        buffer[position] = exp;
    }
    
    position = (position + 1) % capacity;
}

Batch ReplayBuffer::sample(int batch_size) {
    Batch batch;
    batch.size = std::min(static_cast<int>(current_size), batch_size);
    
    if (batch.size == 0) {
        return batch;
    }
    
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    
    batch.states.reserve(batch.size);
    batch.actions.reserve(batch.size);
    batch.next_states.reserve(batch.size);
    batch.rewards.reserve(batch.size);
    batch.dones.reserve(batch.size);
    
    for (int i = 0; i < batch.size; ++i) {
        size_t idx = dist(rng);
        const auto& exp = buffer[idx];
        
        batch.states.push_back(exp.state);
        batch.actions.push_back(exp.action);
        batch.next_states.push_back(exp.next_state);
        batch.rewards.push_back(exp.reward);
        batch.dones.push_back(exp.done);
    }
    
    return batch;
} 