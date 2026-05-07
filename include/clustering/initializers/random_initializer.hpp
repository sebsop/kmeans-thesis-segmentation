/**
 * @file random_initializer.hpp
 * @brief Simple uniform random centroid initialization.
 */

#pragma once

#include <random>

#include "clustering/initializers/initializer.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @class RandomInitializer
 * @brief Implements the simplest K-Means initialization strategy.
 *
 * This class selects 'k' unique points from the input dataset by choosing
 * indices based on a uniform distribution. While fast, it may lead to
 * slower convergence or poor clustering results if the random points are
 * poorly distributed.
 */
class RandomInitializer final : public Initializer {
  public:
    RandomInitializer() = default;
    ~RandomInitializer() override = default;

    /**
     * @brief Randomly selects 'k' points from the input samples.
     * @param samples Input data matrix.
     * @param k Number of centroids to select.
     * @return Vector of randomly selected centroids.
     */
    [[nodiscard]] std::vector<FeatureVector> initialize(const cv::Mat& samples, int k) const override final;
};

} // namespace kmeans::clustering
