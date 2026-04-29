#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"

namespace kmeans::clustering::metrics {

struct BenchmarkResults {
    float wcss{0.0f};            // Within-Cluster Sum of Squares (Inertia)
    float daviesBouldin{0.0f};   // Cluster separation index
    float silhouetteScore{0.0f}; // Approximated silhouette coefficient [-1, 1]
    int iterations{0};           // Number of iterations to converge
    float executionTimeMs{0.0f}; // Total ms taken by the algorithm
};

/**
 * @brief Computes comprehensive clustering quality metrics for offline benchmarking.
 * @param samples The full dataset of samples (N x 5)
 * @param centers The final cluster centroids
 * @param iterations The number of iterations the algorithm took
 * @param executionTimeMs The total time the algorithm took
 */
[[nodiscard]] BenchmarkResults computeAllMetrics(const cv::Mat& samples,
                                                 const std::vector<FeatureVector>& centers,
                                                 int iterations, float executionTimeMs);

} // namespace kmeans::clustering::metrics
