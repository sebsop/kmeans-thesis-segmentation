/**
 * @file classical_engine.hpp
 * @brief GPU implementation of standard K-Means clustering.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/base_kmeans_engine.hpp"

namespace kmeans::clustering {

/**
 * @class ClassicalEngine
 * @brief Standard GPU-accelerated K-Means engine.
 *
 * This engine implements the classical Lloyd's algorithm assignment step on the GPU.
 * It uses standard Euclidean distance metrics to assign each data point to the
 * nearest centroid. It inherits common orchestration from BaseKMeansEngine.
 */
class ClassicalEngine final : public BaseKMeansEngine<ClassicalEngine> {
  public:
    ClassicalEngine() = default;
    ~ClassicalEngine() override = default;

    /**
     * @brief Launches the classical Euclidean-distance assignment kernel.
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