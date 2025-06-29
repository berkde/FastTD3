#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include "replay_buffer_simple.hpp"
#include "normalizers_simple.hpp"
#include "utils_simple.hpp"

TEST(ReplayBufferTest, BasicOperations) {
    ReplayBuffer buffer(1000, 4, 2);
    
    // Test adding samples
    float state[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float action[2] = {0.5f, -0.5f};
    float next_state[4] = {1.1f, 2.1f, 3.1f, 4.1f};
    
    buffer.add(state, action, next_state, 1.0f, false);
    EXPECT_EQ(buffer.size(), 1);
    
    // Test sampling
    auto batch = buffer.sample(1);
    EXPECT_EQ(batch.size, 1);
}

TEST(NormalizerTest, BasicOperations) {
    EmpiricalNormalization normalizer({4}, 1e-8f, 100);
    
    // Test normalization
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto normalized = normalizer.normalize(data);
    
    EXPECT_EQ(normalized.size(), 4);
    
    // Test update
    normalizer.update(data);
    EXPECT_EQ(normalizer.get_mean().size(), 4);
}

TEST(UtilsTest, BasicFunctions) {
    // Test clip function
    EXPECT_EQ(fast_td3::clip(5.0f, 0.0f, 1.0f), 1.0f);
    EXPECT_EQ(fast_td3::clip(-1.0f, 0.0f, 1.0f), 0.0f);
    EXPECT_EQ(fast_td3::clip(0.5f, 0.0f, 1.0f), 0.5f);
    
    // Test TD error
    EXPECT_EQ(fast_td3::compute_td_error(1.0f, 2.0f), 1.0f);
    EXPECT_EQ(fast_td3::compute_td_error(2.0f, 1.0f), -1.0f);
    
    // Test Huber loss
    EXPECT_GT(fast_td3::huber_loss(0.5f, 1.0f), 0.0f);
    EXPECT_EQ(fast_td3::huber_loss(0.0f, 1.0f), 0.0f);
}

TEST(UtilsTest, SoftUpdate) {
    std::vector<float> target = {1.0f, 2.0f, 3.0f};
    std::vector<float> source = {2.0f, 3.0f, 4.0f};
    float tau = 0.1f;
    
    auto result = fast_td3::soft_update(target, source, tau);
    
    EXPECT_EQ(result.size(), 3);
    EXPECT_NEAR(result[0], 1.1f, 1e-6f);
    EXPECT_NEAR(result[1], 2.1f, 1e-6f);
    EXPECT_NEAR(result[2], 3.1f, 1e-6f);
}

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::warn);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 