#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/base_kmeans_engine.hpp"

namespace kmeans::clustering {

class [[deprecated("Use QuantumEngine for enhanced zero-copy pipeline executions.")]] ClassicalEngine final : public BaseKMeansEngine<ClassicalEngine> {
  public:
    ClassicalEngine() = default;
    ~ClassicalEngine() override = default;

    void preRunSetupImpl(const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& /*initialCenters*/,
                         const cv::Mat& /*samples*/) {}

    void launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels, int* d_changed,
                                int threadsPerBlock, int blocksPerGrid, size_t sharedSize);
};

} // namespace kmeans::clustering