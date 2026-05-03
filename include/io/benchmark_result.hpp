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
    std::vector<FeatureVector> classicalCenters{};
    std::vector<FeatureVector> quantumCenters{};
};

} // namespace kmeans::io
