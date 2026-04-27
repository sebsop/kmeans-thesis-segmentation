#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/base_kmeans_engine.hpp"

namespace kmeans::clustering {

class ClassicalEngine final : public BaseKMeansEngine {
  public:
    ClassicalEngine() = default;
    ~ClassicalEngine() override = default;

  protected:
    void launchAssignKernel(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels, int* d_changed,
                            int threadsPerBlock, int blocksPerGrid, size_t sharedSize) override;
};

} // namespace kmeans::clustering