#pragma once

#include <random>

#include "clustering/initializers/initializer.hpp"

namespace kmeans::clustering {

class RandomInitializer final : public Initializer {
  public:
    RandomInitializer() = default;
    ~RandomInitializer() override = default;

    [[nodiscard]] std::vector<cv::Vec<float, 5>> initialize(const cv::Mat& samples, int k) const override final;
};

} // namespace kmeans::clustering
