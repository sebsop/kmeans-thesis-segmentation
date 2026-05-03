#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>
#include <span>

#include "clustering/engines/classical_engine.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Engines {

using namespace kmeans;
using namespace kmeans::clustering;

class Clustering_BaseEngine : public ::testing::Test {
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

// 1. Simple Convergence Test
TEST_F(Clustering_BaseEngine, SimpleConvergence) {
    ClassicalEngine engine;
    
    // 4 points: 2 at (0,0,0,0,0), 2 at (1,1,1,1,1)
    cv::Mat samples(4, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 2; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) samples.at<float>(i, d) = 0.0f;
    }
    for (int i = 2; i < 4; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) samples.at<float>(i, d) = 1.0f;
    }

    // Initial centers: slightly off
    std::vector<FeatureVector> initialCenters(2);
    initialCenters[0] = FeatureVector(0.1f, 0.1f, 0.1f, 0.1f, 0.1f);
    initialCenters[1] = FeatureVector(0.9f, 0.9f, 0.9f, 0.9f, 0.9f);

    auto finalCenters = engine.run(samples, initialCenters, 2, 10);

    // Verify centers moved to exactly 0.0 and 1.0
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(finalCenters[0][d], 0.0f, constants::math::EPSILON);
        EXPECT_NEAR(finalCenters[1][d], 1.0f, constants::math::EPSILON);
    }
}

// 2. Buffer Resizing Test (ensureBuffers)
TEST_F(Clustering_BaseEngine, BufferResizing) {
    ClassicalEngine engine;
    
    // Run small
    cv::Mat small(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0));
    std::vector<FeatureVector> c(2, FeatureVector(0,0,0,0,0));
    EXPECT_NO_THROW(engine.run(small, c, 2, 1));

    // Run large
    cv::Mat large(1000, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0));
    EXPECT_NO_THROW(engine.run(large, c, 2, 1));
}

// 3. Empty Cluster Handling (Random Re-init)
TEST_F(Clustering_BaseEngine, EmptyClusterHandling) {
    ClassicalEngine engine;
    
    // 10 points all at 0.0
    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.0f));
    
    // 2 Centers: both starting far away
    std::vector<FeatureVector> initialCenters(2);
    initialCenters[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    initialCenters[1] = FeatureVector(100.0f, 100.0f, 100.0f, 100.0f, 100.0f); // Will attract 0 points

    auto finalCenters = engine.run(samples, initialCenters, 2, 5);

    // Final centers should both be at 0.0 because the second one should have re-initialized to a point (which is 0.0)
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(finalCenters[0][d], 0.0f, constants::math::EPSILON);
        EXPECT_NEAR(finalCenters[1][d], 0.0f, constants::math::EPSILON);
    }
}

// 4. Run On Device
TEST_F(Clustering_BaseEngine, RunOnDevice) {
    ClassicalEngine engine;
    const int N = 100;
    const int K = 2;
    
    float* d_samples;
    cudaMalloc(&d_samples, N * constants::clustering::FEATURE_DIMS * sizeof(float));
    cudaMemset(d_samples, 0, N * constants::clustering::FEATURE_DIMS * sizeof(float));

    std::vector<FeatureVector> initialCenters(K, FeatureVector(0,0,0,0,0));
    
    EXPECT_NO_THROW({
        auto results = engine.runOnDevice(d_samples, N, initialCenters, K, 5);
        EXPECT_EQ(results.size(), K);
    });

    cudaFree(d_samples);
}

// 5. Early Exit Test (Convergence)
TEST_F(Clustering_BaseEngine, EarlyExitOnConvergence) {
    ClassicalEngine engine;
    
    // Points already perfectly clustered
    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.0f));
    std::vector<FeatureVector> perfectCenters(2);
    perfectCenters[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    perfectCenters[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    engine.run(samples, perfectCenters, 2, 100);
    
    // It should exit after the second iteration because the first one transition from -1 to 0
    EXPECT_EQ(engine.getLastIterations(), 2);
}

// 6. Max Iterations Cap
TEST_F(Clustering_BaseEngine, RespectsMaxIterations) {
    ClassicalEngine engine;
    
    // Create points that would take many iterations to converge
    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(2, FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f));

    const int MAX_ITER = 3;
    engine.run(samples, centers, 2, MAX_ITER);
    
    EXPECT_LE(engine.getLastIterations(), MAX_ITER);
}

// 7. Large Scale Stability
TEST_F(Clustering_BaseEngine, LargeScaleStability) {
    ClassicalEngine engine;
    const int N = 50000;
    const int K = 5;
    
    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(K, FeatureVector(0.5f, 0.5f, 0.5f, 0.5f, 0.5f));

    EXPECT_NO_THROW(engine.run(samples, centers, K, 2));
}

} // namespace ThesisTests::Clustering::Engines
