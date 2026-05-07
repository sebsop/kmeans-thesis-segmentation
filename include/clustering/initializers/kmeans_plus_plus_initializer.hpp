/**
 * @file kmeans_plus_plus_initializer.hpp
 * @brief K-Means++ smart centroid initialization.
 */

#pragma once

#include "clustering/initializers/initializer.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @class KMeansPlusPlusInitializer
 * @brief Implements the K-Means++ initialization algorithm.
 *
 * K-Means++ is a smart initialization technique that spreads out the initial
 * centroids to ensure they are not too close to each other. It chooses the
 * first center randomly and then chooses subsequent centers with probability
 * proportional to their squared distance from the nearest existing center.
 *
 * This generally leads to significantly faster convergence and more stable
 * clustering results compared to simple random selection.
 */
class KMeansPlusPlusInitializer final : public Initializer {
  public:
    KMeansPlusPlusInitializer() = default;
    ~KMeansPlusPlusInitializer() override = default;

    /**
     * @brief Performs the K-Means++ selection process.
     * @param samples Input data matrix.
     * @param k Number of centroids to initialize.
     * @return Vector of K-Means++ selected centroids.
     */
    [[nodiscard]] std::vector<FeatureVector> initialize(const cv::Mat& samples, int k) const override final;
};

} // namespace kmeans::clustering
