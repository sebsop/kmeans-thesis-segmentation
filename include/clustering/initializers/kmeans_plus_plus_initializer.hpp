#pragma once

#include "clustering/initializers/initializer.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

class KMeansPlusPlusInitializer final : public Initializer {
  public:
    KMeansPlusPlusInitializer() = default;
    ~KMeansPlusPlusInitializer() override = default;

    [[nodiscard]] std::vector<FeatureVector> initialize(const cv::Mat& samples, int k) const override final;
};

} // namespace kmeans::clustering
