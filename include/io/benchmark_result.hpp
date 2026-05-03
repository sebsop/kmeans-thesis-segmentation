#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/metrics.hpp"
#include "common/constants.hpp"

namespace kmeans::io {

struct BenchmarkComparisonResult {
    cv::Mat originalFrame{};
    cv::Mat classicalSegmented{};
    cv::Mat quantumSegmented{};
    clustering::metrics::BenchmarkResults classicalMetrics{};
    clustering::metrics::BenchmarkResults quantumMetrics{};
    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> classicalCenters{};
    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> quantumCenters{};
};

} // namespace kmeans::io
