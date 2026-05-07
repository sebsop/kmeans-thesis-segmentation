/**
 * @file benchmark_result.hpp
 * @brief Data structures for comparative analysis reports.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/metrics.hpp"
#include "common/constants.hpp"

namespace kmeans::io {

/**
 * @struct BenchmarkComparisonResult
 * @brief Aggregates the outcome of a head-to-head algorithm comparison.
 *
 * This structure is the primary data transfer object between the
 * BenchmarkRunner and the UI. It contains everything needed to render
 * a side-by-side comparison of the Classical and Quantum implementations.
 */
struct BenchmarkComparisonResult {
    cv::Mat originalFrame{};      ///< The frozen frame used for the test
    cv::Mat classicalSegmented{}; ///< Resulting image from the Classical engine
    cv::Mat quantumSegmented{};   ///< Resulting image from the Quantum engine

    clustering::metrics::BenchmarkResults classicalMetrics{}; ///< Accuracy/Performance metrics for Classical
    clustering::metrics::BenchmarkResults quantumMetrics{};   ///< Accuracy/Performance metrics for Quantum

    std::vector<FeatureVector> classicalCenters{}; ///< Final centroid coordinates for Classical
    std::vector<FeatureVector> quantumCenters{};   ///< Final centroid coordinates for Quantum
};

} // namespace kmeans::io
