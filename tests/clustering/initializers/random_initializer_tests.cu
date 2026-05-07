/**
 * @file random_initializer_tests.cu
 * @brief Unit tests for the uniform random centroid initialization strategy.
 */

#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/initializers/random_initializer.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Initializers {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying random initialization logic.
 */
class Initializer_Random : public ::testing::Test {};

/**
 * @brief Verifies that the initializer returns exactly the requested number of centroids.
 */
TEST_F(Initializer_Random, ReturnsKCenters) {
    RandomInitializer init;
    const int K = 5;
    const int N = 100;

    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.0f));
    auto centers = init.initialize(samples, K);

    EXPECT_EQ(centers.size(), K);
}

/**
 * @brief Ensures that initialized centroids remain within the spatial and color boundaries of the input data.
 */
TEST_F(Initializer_Random, CentersWithinInputRange) {
    RandomInitializer init;
    const int N = 10;
    cv::Mat samples(N, constants::clustering::FEATURE_DIMS, CV_32F);

    // Set data in range [0.4, 0.6]
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 0.4f + (i * 0.01f);
        }
    }

    auto centers = init.initialize(samples, 3);

    for (const auto& c : centers) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            EXPECT_GE(c[d], 0.4f);
            EXPECT_LE(c[d], 0.6f);
        }
    }
}

/**
 * @brief Verifies that the internal RNG is active and produces non-constant results.
 */
TEST_F(Initializer_Random, IsNotConstant) {
    RandomInitializer init;
    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 100; ++i) {
        samples.at<float>(i, 0) = static_cast<float>(i);
    }

    auto centers1 = init.initialize(samples, 1);
    auto centers2 = init.initialize(samples, 1);

    // Verifies the RNG state advances between calls
    EXPECT_NE(centers1[0][0], centers2[0][0]);
}

} // namespace ThesisTests::Clustering::Initializers
