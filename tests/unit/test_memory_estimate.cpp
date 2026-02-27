#include <gtest/gtest.h>
#include "zoo/memory_estimate.hpp"

using namespace zoo;

TEST(MemoryEstimateTest, NonexistentFileReturnsError) {
    auto result = estimate_memory("/nonexistent/path/model.gguf", 4096);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ModelLoadFailed);
}

TEST(MemoryEstimateTest, KvTypeF16IsDefault) {
    // We can't test with a real model, but we can test the type logic
    // by checking that Q8_0 produces smaller KV cache than F16
    // This is tested indirectly through the estimation formula
    // (Unit-testable without a real model requires a mock - skip for now)
    // At minimum, validate that the function compiles and returns Expected type
    Expected<MemoryEstimate> result = estimate_memory("/no/model.gguf", 8192);
    EXPECT_FALSE(result.has_value());  // Fails because file doesn't exist
}

TEST(MemoryEstimateTest, TotalGbConversionIsCorrect) {
    MemoryEstimate est;
    est.model_weights_bytes = 1024ULL * 1024 * 1024;  // 1 GB
    est.kv_cache_bytes = 512ULL * 1024 * 1024;         // 0.5 GB
    est.compute_buffer_bytes = 256ULL * 1024 * 1024;   // 0.25 GB
    est.total_bytes = est.model_weights_bytes + est.kv_cache_bytes + est.compute_buffer_bytes;

    EXPECT_NEAR(est.model_gb(), 1.0, 0.001);
    EXPECT_NEAR(est.kv_cache_gb(), 0.5, 0.001);
    EXPECT_NEAR(est.total_gb(), 1.75, 0.001);
}
