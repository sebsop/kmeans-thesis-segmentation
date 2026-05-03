#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/base_kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

class QuantumEngine final : public BaseKMeansEngine<QuantumEngine> {
  private:
    float m_scaleFactor = constants::quantum::SCALE_FACTOR;

  public:
    QuantumEngine() = default;
    ~QuantumEngine() override = default;

    void launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels, int* d_changed,
                                int threadsPerBlock, int blocksPerGrid, size_t sharedSize);
};

} // namespace kmeans::clustering
