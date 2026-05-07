/**
 * @file base_kmeans_engine_tests.cu
 * @brief Unit tests for the core K-Means iterative logic and GPU buffer management.
 */

#include <cuda_runtime.h>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/engines/classical_engine.hpp"
#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Engines {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying the fundamental clustering loop.
 */
class Clustering_BaseEngine : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            FAIL() << "No CUDA-capable GPU detected.";
        }
    }

    void SetUp() override { cudaDeviceReset(); }
};

/**
 * @brief Verifies that the engine converges to the global minimum on a simple linearly separable dataset.
 */
TEST_F(Clustering_BaseEngine, SimpleConvergence) {
    ClassicalEngine engine;

    // 4 points: 2 at (0,0,0,0,0), 2 at (1,1,1,1,1)
    cv::Mat samples(4, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 2; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 0.0f;
        }
    }
    for (int i = 2; i < 4; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 1.0f;
        }
    }

    // Initial centers: intentionally slightly off-center
    std::vector<FeatureVector> initialCenters(2);
    initialCenters[0] = FeatureVector(0.1f, 0.1f, 0.1f, 0.1f, 0.1f);
    initialCenters[1] = FeatureVector(0.9f, 0.9f, 0.9f, 0.9f, 0.9f);

    auto finalCenters = engine.run(samples, initialCenters, 2, 10);

    // Verify centers moved to exact centroids
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(finalCenters[0][d], 0.0f, constants::math::EPSILON);
        EXPECT_NEAR(finalCenters[1][d], 1.0f, constants::math::EPSILON);
    }
}

/**
 * @brief Verifies the stability of the dynamic GPU buffer resizing mechanism.
 */
TEST_F(Clustering_BaseEngine, BufferResizing) {
    ClassicalEngine engine;

    // Small scale
    cv::Mat small(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0));
    std::vector<FeatureVector> c(2, FeatureVector(0, 0, 0, 0, 0));
    EXPECT_NO_THROW((void)engine.run(small, c, 2, 1));

    // Large scale (forces realloc)
    cv::Mat large(1000, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0));
    EXPECT_NO_THROW((void)engine.run(large, c, 2, 1));
}

/**
 * @brief Validates the early-exit optimization when convergence is detected.
 */
TEST_F(Clustering_BaseEngine, EarlyExitOnConvergence) {
    ClassicalEngine engine;

    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.0f));
    std::vector<FeatureVector> perfectCenters(2);
    perfectCenters[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    perfectCenters[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    (void)engine.run(samples, perfectCenters, 2, 100);

    // Verify the engine correctly identifies the stable state
    EXPECT_EQ(engine.getLastIterations(), 2);
}

/**
 * @brief Ensures the iterative loop strictly respects the provided MAX_ITERations cap.
 */
TEST_F(Clustering_BaseEngine, RespectsMaxIterations) {
    ClassicalEngine engine;

    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(2, FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f));

    const int MAX_ITER = 3;
    (void)engine.run(samples, centers, 2, MAX_ITER);

    EXPECT_LE(engine.getLastIterations(), MAX_ITER);
}

/**
 * @brief Stress tests the memory management during rapid resolution/K-count changes.
 */
TEST_F(Clustering_BaseEngine, HostileMemoryReallocation) {
    ClassicalEngine engine;

    for (int i = 1; i <= 5; ++i) {
        int N = i * 1000;
        int K = i * 2;
        cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
        std::vector<FeatureVector> centers(K, FeatureVector(0.5f, 0.5f, 0.5f, 0.5f, 0.5f));

        EXPECT_NO_THROW((void)engine.run(samples, centers, K, 2));
    }
}

/**
 * @brief Verifies stability when processing zero-point input sets.
 */
TEST_F(Clustering_BaseEngine, HandlesZeroPointsGracefully) {
    ClassicalEngine engine;
    cv::Mat emptySamples;
    std::vector<FeatureVector> centers(3, FeatureVector(0, 0, 0, 0, 0));

    EXPECT_NO_THROW({
        auto results = engine.run(emptySamples, centers, 3, 5);
        EXPECT_EQ(results.size(), 3);
    });
}

} // namespace ThesisTests::Clustering::Engines
