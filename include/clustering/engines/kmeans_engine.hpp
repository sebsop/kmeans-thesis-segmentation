/**
 * @file kmeans_engine.hpp
 * @brief Strategy Pattern interface for K-Means execution backends.
 */

#pragma once

#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"
#include "cuda_runtime.h"

namespace kmeans::clustering {

/**
 * @class KMeansEngine
 * @brief Abstract base class defining the contract for K-Means implementations.
 *
 * This interface follows the Strategy Design Pattern, allowing the application to
 * swap between different computational backends (e.g., standard CPU-based Lloyds
 * algorithm vs. Quantum-assisted sampling) at runtime without modifying the
 * high-level clustering logic.
 */
class KMeansEngine {
  protected:
    int m_lastIterations = 0; ///< Number of iterations performed in the last execution

  public:
    virtual ~KMeansEngine() = default;

    /**
     * @brief Gets the number of iterations from the most recent run.
     * @return Integer count of iterations.
     */
    [[nodiscard]] int getLastIterations() const noexcept { return m_lastIterations; }

    /**
     * @brief Executes clustering on CPU-resident data.
     *
     * Implementations of this method should handle the data provided in the
     * OpenCV Mat, perform the clustering logic, and return the final centroids.
     *
     * @param samples Input data matrix where each row is a sample.
     * @param initialCenters The starting centroids to begin optimization.
     * @param k Target number of clusters.
     * @param maxIterations Hard limit on the number of optimization steps.
     * @return A vector of the finalized FeatureVector centroids.
     */
    [[nodiscard]] virtual std::vector<FeatureVector>
    run(const cv::Mat& samples, std::span<const FeatureVector> initialCenters, int k, int maxIterations) = 0;

    /**
     * @brief Executes clustering on GPU-resident data (Direct Path).
     *
     * This "GPU-Direct" path is an optimization that skips the Host-to-Device
     * transfer of sample data. It is intended for scenarios where the data
     * preprocessing already occurred on the GPU.
     *
     * @param d_samples_ext Pointer to raw float array in VRAM.
     * @param numPoints Total number of samples in the buffer.
     * @param initialCenters The starting centroids.
     * @param k Target number of clusters.
     * @param maxIterations Optimization limit.
     * @return A vector of the finalized FeatureVector centroids.
     */
    [[nodiscard]] virtual std::vector<FeatureVector> runOnDevice(float* /*d_samples_ext*/, int /*numPoints*/,
                                                                 std::span<const FeatureVector> /*initialCenters*/,
                                                                 int /*k*/, int /*maxIterations*/) {
        // Default implementation returns empty; subclasses must override to enable GPU path.
        return {};
    }
};

} // namespace kmeans::clustering
