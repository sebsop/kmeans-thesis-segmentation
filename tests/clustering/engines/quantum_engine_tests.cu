#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>

#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Engines {

using namespace kmeans;
using namespace kmeans::clustering;

class Clustering_QuantumEngine : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            FAIL() << "No CUDA-capable GPU detected.";
        }
    }

    void SetUp() override {
        cudaDeviceReset();
    }
};

// 1. Quantum Interference Approximation Test
TEST_F(Clustering_QuantumEngine, QuantumMetricAssignment) {
    QuantumEngine engine;
    
    // Points and Centers at the exact same location
    // The quantum probability should be 1.0 (cos(0)=1), so distance should be 0.0
    cv::Mat samples(1, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.75f));
    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.75f, 0.75f, 0.75f, 0.75f, 0.75f); // Exact match
    centers[1] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);     // Far away

    auto results = engine.run(samples, centers, 2, 2);
    
    // P0 should be assigned to C0, making the new C0 = 0.75
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(results[0][d], 0.75f, constants::math::EPSILON);
    }
}

// 2. Convergence with Quantum Metric
TEST_F(Clustering_QuantumEngine, QuantumConvergence) {
    QuantumEngine engine;
    
    // 2 clusters far apart
    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 5; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) samples.at<float>(i, d) = 0.1f;
    }
    for (int i = 5; i < 10; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) samples.at<float>(i, d) = 0.9f;
    }

    std::vector<FeatureVector> initialCenters(2);
    initialCenters[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    initialCenters[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    auto finalCenters = engine.run(samples, initialCenters, 2, 10);

    // Verify it converges to the centers (0.1 and 0.9)
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(finalCenters[0][d], 0.1f, 0.01f);
        EXPECT_NEAR(finalCenters[1][d], 0.9f, 0.01f);
    }
}

// 3. Shared Memory Stress (similar to classical but with quantum-specific shared size)
TEST_F(Clustering_QuantumEngine, QuantumHighK) {
    QuantumEngine engine;
    const int K = constants::clustering::K_MAX;
    
    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(K, FeatureVector(0.5f, 0.5f, 0.5f, 0, 0));

    EXPECT_NO_THROW(engine.run(samples, centers, K, 2));
}

} // namespace ThesisTests::Clustering::Engines
