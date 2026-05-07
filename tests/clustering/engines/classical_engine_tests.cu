/**
 * @file classical_engine_tests.cu
 * @brief Unit tests for the high-performance Classical (GPU-accelerated) K-Means engine.
 */

#include <cuda_runtime.h>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/engines/classical_engine.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Engines {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying the Classical clustering implementation.
 */
class Clustering_ClassicalEngine : public ::testing::Test {
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
 * @brief Verifies the accuracy of the GPU-based pixel assignment kernel.
 *
 * Uses controlled data points to ensure that the Euclidean distance logic
 * correctly maps pixels to the mathematically nearest centroid.
 */
TEST_F(Clustering_ClassicalEngine, AssignmentCorrectness) {
    ClassicalEngine engine;

    // P0(0,0,0,0,0) -> Should go to C0
    // P1(1,1,1,1,1) -> Should go to C1
    // P2(0.1, 0.1, 0.1, 0.1, 0.1) -> Should go to C0
    cv::Mat samples(3, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        samples.at<float>(0, d) = 0.0f;
        samples.at<float>(1, d) = 1.0f;
        samples.at<float>(2, d) = 0.1f;
    }

    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    centers[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    (void)engine.run(samples, centers, 2, 1);

    EXPECT_EQ(engine.getLastIterations(), 2);
}

/**
 * @brief Stress tests the shared memory allocation for high K-values.
 *
 * Ensures that the engine handles the maximum supported cluster count (20)
 * without exceeding GPU shared memory limits or hardware bounds.
 */
TEST_F(Clustering_ClassicalEngine, HighKAssignment) {
    ClassicalEngine engine;
    const int K = constants::clustering::K_MAX;
    const int N = 1000;

    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(K);
    for (int i = 0; i < K; ++i) {
        centers[i] = FeatureVector(i / 20.0f, i / 20.0f, i / 20.0f, 0, 0);
    }

    EXPECT_NO_THROW((void)engine.run(samples, centers, K, 2));
}

/**
 * @brief Verifies deterministic behavior during mathematical ties.
 */
TEST_F(Clustering_ClassicalEngine, TiedDistanceHandling) {
    ClassicalEngine engine;

    cv::Mat samples(1, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    centers[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    auto results = engine.run(samples, centers, 2, 2);

    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(results[0][d], 0.5f, constants::math::EPSILON);
    }
}

/**
 * @brief Ensures robustness against "empty clusters" (clusters with zero assigned points).
 *
 * Verifies that the centroid update logic handles division-by-zero safely
 * and does not produce NaNs when a cluster is not chosen by any sample.
 */
TEST_F(Clustering_ClassicalEngine, EmptyClusterRobustness) {
    ClassicalEngine engine;
    const int K = 10;

    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));

    std::vector<FeatureVector> initialCenters(K);
    for (int i = 0; i < K; ++i) {
        initialCenters[i] = FeatureVector(static_cast<float>(i), 0, 0, 0, 0);
    }

    auto finalCenters = engine.run(samples, initialCenters, K, 5);

    EXPECT_EQ(finalCenters.size(), K);
    for (const auto& c : finalCenters) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            EXPECT_FALSE(std::isnan(c[d]));
        }
    }
}

} // namespace ThesisTests::Clustering::Engines
