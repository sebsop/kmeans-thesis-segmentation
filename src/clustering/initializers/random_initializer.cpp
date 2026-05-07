/**
 * @file random_initializer.cpp
 * @brief Implementation of the uniform random sampling initialization.
 */

#include "clustering/initializers/random_initializer.hpp"

#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @brief Selects K random points from the input data to serve as initial centroids.
 *
 * This implementation uses a thread-safe random number generator (std::mt19937)
 * to pick K indices from the preprocessed feature matrix. Each selected row
 * is copied into a FeatureVector.
 *
 * @note This method is computationally inexpensive but can lead to poor
 *       convergence if the initial points are too close together. It serves
 *       as the standard baseline for K-Means.
 *
 * @param samples Preprocessed 5D feature matrix.
 * @param k Number of clusters to initialize.
 * @return std::vector<FeatureVector> The K selected starting points.
 */
std::vector<FeatureVector> RandomInitializer::initialize(const cv::Mat& samples, int k) const {
    std::vector<FeatureVector> centers(k);
    int numPoints = samples.rows;

    // Thread-local RNG to ensure efficiency and safety in the multithreaded app
    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    // Uniformly sample K points from the image
    std::ranges::generate(centers, [&]() {
        int randIdx = dis(gen);
        const auto* rowPtr = samples.ptr<float>(randIdx);
        FeatureVector c;
        std::copy_n(rowPtr, constants::clustering::FEATURE_DIMS, c.val);
        return c;
    });

    return centers;
}

} // namespace kmeans::clustering
