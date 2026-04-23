#pragma once

#include <memory>
#include <vector>

#include "clustering/engines/kmeans_engine.hpp"

namespace kmeans::clustering {

class QuantumEngine final : public KMeansEngine {
  public:
    QuantumEngine() = default;
    ~QuantumEngine() = default;

    [[nodiscard]] std::vector<cv::Vec<float, 5>>
    run(const cv::Mat& samples, const std::vector<cv::Vec<float, 5>>& initialCenters, int k) override final;
};

} // namespace kmeans::clustering
