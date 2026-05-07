/**
 * @file initializer.hpp
 * @brief Strategy Pattern interface for cluster centroid initialization.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @class Initializer
 * @brief Abstract base class for different K-Means initialization strategies.
 *
 * This interface follows the Strategy Design Pattern, allowing the application
 * to choose how the initial centroids are selected before the main K-Means
 * iteration begins. Proper initialization is critical for algorithm
 * convergence speed and final clustering quality.
 */
class Initializer {
  public:
    virtual ~Initializer() = default;

    /**
     * @brief Computes the starting centroids for the K-Means algorithm.
     *
     * Implementations should select 'k' representative points from the
     * input 'samples' to serve as the initial cluster centers.
     *
     * @param samples Input data matrix (cv::Mat).
     * @param k The number of clusters to initialize.
     * @return A vector of 'k' initialized FeatureVector centroids.
     */
    [[nodiscard]] virtual std::vector<FeatureVector> initialize(const cv::Mat& samples, int k) const = 0;
};

} // namespace kmeans::clustering
