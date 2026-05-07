/**
 * @file metrics_tests.cpp
 * @brief Unit tests for clustering quality metrics (Silhouette, Davies-Bouldin, WCSS).
 */

#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/metrics.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying mathematical clustering metrics.
 */
class Clustering_Metrics : public ::testing::Test {};

/**
 * @brief Verifies that the Silhouette Score approaches 1.0 for perfectly separated data.
 *
 * Also checks that WCSS (Within-Cluster Sum of Squares) is near zero when
 * samples align perfectly with their centroids.
 */
TEST_F(Clustering_Metrics, SilhouetteOnPerfectSeparation) {
    // 2 clusters at (0,0,0,0,0) and (1,1,1,1,1)
    cv::Mat samples(20, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 10; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 0.0f;
        }
    }
    for (int i = 10; i < 20; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 1.0f;
        }
    }

    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    centers[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    auto results = metrics::computeAllMetrics(samples, centers, 1, 0.1f);

    EXPECT_GT(results.silhouetteScore, 0.9f);
    EXPECT_NEAR(results.wcss, 0.0f, 1e-5);
}

/**
 * @brief Verifies that the Silhouette Score penalizes ambiguous or overlapping clusters.
 */
TEST_F(Clustering_Metrics, SilhouetteOnOverlappingData) {
    // Overlapping points assigned to different centers
    cv::Mat samples(20, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));

    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.5f, 0.5f, 0.5f, 0.5f, 0.5f);
    centers[1] = FeatureVector(0.51f, 0.51f, 0.51f, 0.51f, 0.51f);

    auto results = metrics::computeAllMetrics(samples, centers, 1, 0.1f);

    // Score should be significantly lower than the ideal case
    EXPECT_LT(results.silhouetteScore, 0.3f);
}

/**
 * @brief Validates the Davies-Bouldin Index behavior for different cluster densities.
 *
 * Verifies that the DB index is lower (indicating better clustering) when
 * clusters are compact and well-separated.
 */
TEST_F(Clustering_Metrics, DaviesBouldinIndexScaling) {
    // Case A: Compact clusters far apart
    cv::Mat samplesA(20, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 10; ++i) {
        for (int d = 0; d < 5; ++d) {
            samplesA.at<float>(i, d) = 0.0f;
        }
    }
    for (int i = 10; i < 20; ++i) {
        for (int d = 0; d < 5; ++d) {
            samplesA.at<float>(i, d) = 1.0f;
        }
    }

    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0, 0, 0, 0, 0);
    centers[1] = FeatureVector(1, 1, 1, 1, 1);

    auto metricsA = metrics::computeAllMetrics(samplesA, centers, 1, 0.1f);

    // Case B: Broad clusters closer together
    cv::Mat samplesB(20, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 10; ++i) {
        for (int d = 0; d < 5; ++d) {
            samplesB.at<float>(i, d) = 0.45f;
        }
    }
    for (int i = 10; i < 20; ++i) {
        for (int d = 0; d < 5; ++d) {
            samplesB.at<float>(i, d) = 0.55f;
        }
    }

    auto metricsB = metrics::computeAllMetrics(samplesB, centers, 1, 0.1f);

    EXPECT_LT(metricsA.daviesBouldin, metricsB.daviesBouldin);
}

/**
 * @brief Ensures the metrics module handles empty datasets without crashing.
 */
TEST_F(Clustering_Metrics, HandlesZeroData) {
    cv::Mat empty;
    std::vector<FeatureVector> centers;

    auto results = metrics::computeAllMetrics(empty, centers, 0, 0.0f);
    EXPECT_EQ(results.wcss, 0.0f);
    EXPECT_EQ(results.daviesBouldin, 0.0f);
}

} // namespace ThesisTests::Clustering
