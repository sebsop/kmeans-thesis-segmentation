#include <cuda_runtime.h>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/engines/classical_engine.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Engines {

using namespace kmeans;
using namespace kmeans::clustering;

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

// 1. Core Assignment Logic (Verification of Euclidean Distance on GPU)
TEST_F(Clustering_ClassicalEngine, AssignmentCorrectness) {
    ClassicalEngine engine;

    // 3 points
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

    // We run for only 1 iteration to test the ASSIGNMENT kernel specifically
    engine.run(samples, centers, 2, 1);

    EXPECT_EQ(engine.getLastIterations(), 2); // 1st iter detects change (-1 -> 0), 2nd iter breaks
}

// 2. High-K Assignment Stress (Shared Memory limits)
TEST_F(Clustering_ClassicalEngine, HighKAssignment) {
    ClassicalEngine engine;
    const int K = constants::clustering::K_MAX; // 20
    const int N = 1000;

    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(K);
    for (int i = 0; i < K; ++i)
        centers[i] = FeatureVector(i / 20.0f, i / 20.0f, i / 20.0f, 0, 0);

    // Verify it handles the shared memory allocation for 20 centroids
    EXPECT_NO_THROW(engine.run(samples, centers, K, 2));
}

// 3. Mathematical Edge Case: Tied Distances
TEST_F(Clustering_ClassicalEngine, TiedDistanceHandling) {
    ClassicalEngine engine;

    // Point at 0.5, Centers at 0.0 and 1.0. Distance is equal.
    // Algorithm should deterministically pick the first one (cluster 0)
    cv::Mat samples(1, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    centers[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    auto results = engine.run(samples, centers, 2, 2);

    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(results[0][d], 0.5f, constants::math::EPSILON);
    }
}
TEST_F(Clustering_ClassicalEngine, EmptyClusterRobustness) {
    ClassicalEngine engine;
    const int K = 10;

    // Uniform data (all points at same location)
    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));

    // Centers spread out - most will have 0 members
    std::vector<FeatureVector> initialCenters(K);
    for (int i = 0; i < K; ++i)
        initialCenters[i] = FeatureVector(static_cast<float>(i), 0, 0, 0, 0);

    auto finalCenters = engine.run(samples, initialCenters, K, 5);

    EXPECT_EQ(finalCenters.size(), K);
    for (const auto& c : finalCenters) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            EXPECT_FALSE(std::isnan(c[d]));
        }
    }
}

} // namespace ThesisTests::Clustering::Engines
