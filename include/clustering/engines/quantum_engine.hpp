#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/base_kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

class QuantumEngine final : public BaseKMeansEngine {
  private:
    float m_scaleFactor = 1.0f;

  public:
    QuantumEngine() = default;
    ~QuantumEngine() override = default;

  protected:
    void preRunSetup(const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& initialCenters,
                     const cv::Mat& samples) override;

    void launchAssignKernel(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels, int* d_changed,
                            int threadsPerBlock, int blocksPerGrid, size_t sharedSize) override;
};

} // namespace kmeans::clustering
