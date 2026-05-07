/**
 * @file quantum_engine.hpp
 * @brief Hybrid Quantum-Classical K-Means engine implementation.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/base_kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @class QuantumEngine
 * @brief Experimental K-Means engine utilizing quantum-inspired assignment.
 *
 * This engine serves as the core experimental component of the thesis. It
 * implements a hybrid approach where the point-to-centroid assignment is
 * performed using quantum-assisted distance estimation (simulated on the GPU).
 *
 * By inheriting from BaseKMeansEngine, it integrates seamlessly into the
 * standard clustering pipeline while providing a different mathematical
 * foundation for the optimization step.
 */
class QuantumEngine final : public BaseKMeansEngine<QuantumEngine> {
  public:
    QuantumEngine() = default;
    ~QuantumEngine() override = default;

    /**
     * @brief Launches the quantum-assisted assignment kernel.
     *
     * @param d_samples Pointer to samples in VRAM.
     * @param numPoints Number of samples.
     * @param d_centers Pointer to current centroids in VRAM.
     * @param k Number of clusters.
     * @param d_labels Output buffer for point assignments.
     * @param d_changed Output flag indicating if any labels changed.
     * @param threadsPerBlock CUDA block dimension.
     * @param blocksPerGrid CUDA grid dimension.
     * @param sharedSize Shared memory allocation size in bytes.
     */
    void launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels, int* d_changed,
                                int threadsPerBlock, int blocksPerGrid, size_t sharedSize);
};

} // namespace kmeans::clustering
