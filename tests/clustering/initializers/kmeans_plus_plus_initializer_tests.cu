#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Initializers {

using namespace kmeans;
using namespace kmeans::clustering;

class Initializer_KMeansPlusPlus : public ::testing::Test {
  protected:
    void SetUp() override {
        // No specific setup needed for CPU-based initializer
    }
};

// 1. Basic Functionality: Ensure K distinct centers are returned
TEST_F(Initializer_KMeansPlusPlus, ReturnsKDistinctCenters) {
    KMeansPlusPlusInitializer init;
    const int K = 5;
    const int N = 100;

    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = static_cast<float>(i) / N;
        }
    }

    auto centers = init.initialize(samples, K);

    EXPECT_EQ(centers.size(), K);

    // Check for uniqueness (using a simple set of first dimension for this test)
    std::set<float> uniqueVals;
    for (const auto& c : centers) {
        uniqueVals.insert(c[0]);
    }

    // In this structured dataset, each center should be unique
    EXPECT_EQ(uniqueVals.size(), K);
}

// 2. Probabilistic Separation (The "Plus Plus" part)
// With two very far apart points, it MUST pick both as centers if K=2
TEST_F(Initializer_KMeansPlusPlus, SeparationOnDistantClusters) {
    KMeansPlusPlusInitializer init;
    const int K = 2;

    cv::Mat samples(2, constants::clustering::FEATURE_DIMS, CV_32F);
    // Point A at 0.0
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d)
        samples.at<float>(0, d) = 0.0f;
    // Point B at 100.0 (Far away)
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d)
        samples.at<float>(1, d) = 100.0f;

    auto centers = init.initialize(samples, K);

    EXPECT_EQ(centers.size(), 2);

    float d1 = centers[0][0];
    float d2 = centers[1][0];

    // One must be 0 and other must be 100
    EXPECT_TRUE((std::abs(d1 - 0.0f) < 1e-5 && std::abs(d2 - 100.0f) < 1e-5) ||
                (std::abs(d1 - 100.0f) < 1e-5 && std::abs(d2 - 0.0f) < 1e-5));
}

// 3. Edge Case: K = 1
TEST_F(Initializer_KMeansPlusPlus, HandlesKEqualsOne) {
    KMeansPlusPlusInitializer init;
    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));

    auto centers = init.initialize(samples, 1);
    EXPECT_EQ(centers.size(), 1);
}

// 4. Edge Case: K > N
TEST_F(Initializer_KMeansPlusPlus, HandlesKLargerThanN) {
    KMeansPlusPlusInitializer init;
    const int N = 3;
    const int K = 5;

    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.1f));

    // Should gracefully return at most N centers or repeat them safely
    auto centers = init.initialize(samples, K);
    EXPECT_LE(centers.size(), K);
}

// 5. Stability on Constant Data
TEST_F(Initializer_KMeansPlusPlus, StabilityOnUniformData) {
    KMeansPlusPlusInitializer init;
    cv::Mat samples(50, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.77f));

    auto centers = init.initialize(samples, 3);
    EXPECT_EQ(centers.size(), 3);
    for (const auto& c : centers) {
        EXPECT_NEAR(c[0], 0.77f, 1e-5);
    }
}

} // namespace ThesisTests::Clustering::Initializers
