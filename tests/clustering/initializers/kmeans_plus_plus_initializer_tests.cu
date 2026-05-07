/**
 * @file kmeans_plus_plus_initializer_tests.cu
 * @brief Unit tests for the K-Means++ (probabilistic) initialization strategy.
 */

#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Initializers {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying K-Means++ initialization logic.
 */
class Initializer_KMeansPlusPlus : public ::testing::Test {
  protected:
    void SetUp() override {}
};

/**
 * @brief Verifies that the algorithm selects K distinct centers from a structured dataset.
 */
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

    // Check for uniqueness across the first dimension
    std::set<float> uniqueVals;
    for (const auto& c : centers) {
        uniqueVals.insert(c[0]);
    }

    EXPECT_EQ(uniqueVals.size(), K);
}

/**
 * @brief Validates the "Plus Plus" probabilistic separation logic.
 *
 * In a dataset with two distant points, the algorithm MUST pick both
 * as centers for K=2 due to the D(x)^2 weighting.
 */
TEST_F(Initializer_KMeansPlusPlus, SeparationOnDistantClusters) {
    KMeansPlusPlusInitializer init;
    const int K = 2;

    cv::Mat samples(2, constants::clustering::FEATURE_DIMS, CV_32F);
    // Point A at 0.0
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        samples.at<float>(0, d) = 0.0f;
    }
    // Point B at 100.0 (Mathematically forcing selection)
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        samples.at<float>(1, d) = 100.0f;
    }

    auto centers = init.initialize(samples, K);

    EXPECT_EQ(centers.size(), 2);

    float d1 = centers[0][0];
    float d2 = centers[1][0];

    EXPECT_TRUE((std::abs(d1 - 0.0f) < 1e-5 && std::abs(d2 - 100.0f) < 1e-5) ||
                (std::abs(d1 - 100.0f) < 1e-5 && std::abs(d2 - 0.0f) < 1e-5));
}

/**
 * @brief Verifies stability when K=1.
 */
TEST_F(Initializer_KMeansPlusPlus, HandlesKEqualsOne) {
    KMeansPlusPlusInitializer init;
    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));

    auto centers = init.initialize(samples, 1);
    EXPECT_EQ(centers.size(), 1);
}

/**
 * @brief Verifies robustness when the requested K exceeds the available sample count N.
 */
TEST_F(Initializer_KMeansPlusPlus, HandlesKLargerThanN) {
    KMeansPlusPlusInitializer init;
    const int N = 3;
    const int K = 5;

    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.1f));

    auto centers = init.initialize(samples, K);
    EXPECT_LE(centers.size(), K);
}

/**
 * @brief Verifies stability on uniform datasets where all points are identical.
 */
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
