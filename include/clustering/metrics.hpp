#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace kmeans::clustering::metrics {

struct BenchmarkResults {
    float wcss;            // Within-Cluster Sum of Squares (Inertia)
    float daviesBouldin;   // Cluster separation index
    float silhouetteScore; // Approximated silhouette coefficient [-1, 1]
    int iterations;        // Number of iterations to converge
    float executionTimeMs; // Total ms taken by the algorithm
};

/**
 * @brief Computes comprehensive clustering quality metrics for offline benchmarking.
 * @param samples The full dataset of samples (N x 5)
 * @param centers The final cluster centroids
 * @param iterations The number of iterations the algorithm took
 * @param executionTimeMs The total time the algorithm took
 */
BenchmarkResults computeAllMetrics(const cv::Mat& samples, const std::vector<cv::Vec<float, 5>>& centers,
                                   int iterations, float executionTimeMs);

} // namespace kmeans::clustering::metrics
