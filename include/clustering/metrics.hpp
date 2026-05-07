/**
 * @file metrics.hpp
 * @brief Evaluation metrics for clustering quality and performance.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"

namespace kmeans::clustering::metrics {

/**
 * @struct BenchmarkResults
 * @brief Data structure holding the results of a clustering performance and quality audit.
 */
struct BenchmarkResults {
    /** @brief Within-Cluster Sum of Squares (Inertia). Measures how tightly grouped the clusters are. Lower is better.
     */
    float wcss{0.0f};

    /** @brief Davies-Bouldin Index. Measures the average "similarity" between clusters. Lower is better. */
    float daviesBouldin{0.0f};

    /** @brief Silhouette Score (Approximated). Measures how similar a point is to its own cluster vs others. Range [-1,
     * 1]. Higher is better. */
    float silhouetteScore{0.0f};

    /** @brief Total iterations performed until convergence. */
    int iterations{0};

    /** @brief Total wall-clock time in milliseconds for the clustering operation. */
    float executionTimeMs{0.0f};
};

/**
 * @brief Computes comprehensive clustering quality metrics for offline benchmarking.
 *
 * This function calculates mathematical metrics that quantify how well the
 * algorithm segmented the data. These metrics are essential for the
 * comparative analysis part of the thesis.
 *
 * @param samples The full dataset of samples (N x 5 feature vectors).
 * @param centers The final optimized cluster centroids.
 * @param iterations The number of iterations the algorithm performed.
 * @param executionTimeMs The total time spent in the clustering engine.
 * @return A BenchmarkResults object containing the calculated scores.
 */
[[nodiscard]] BenchmarkResults computeAllMetrics(const cv::Mat& samples, const std::vector<FeatureVector>& centers,
                                                 int iterations, float executionTimeMs);

} // namespace kmeans::clustering::metrics
