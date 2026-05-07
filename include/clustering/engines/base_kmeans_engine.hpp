/**
 * @file base_kmeans_engine.hpp
 * @brief Base implementation for GPU-accelerated K-Means engines using CRTP.
 */

#pragma once

#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @class BaseKMeansEngine
 * @brief A template-based base class providing common CUDA orchestration for K-Means.
 *
 * This class implements the core K-Means iterative loop while delegating the specific
 * "Point Assignment" kernel launch to the derived class via the Curiously Recurring
 * Template Pattern (CRTP). This allows for static polymorphism, avoiding virtual
 * function overhead in the performance-critical inner loop.
 *
 * It manages common GPU buffers for samples, labels, and intermediate sums,
 * ensuring they are lazily allocated and resized only when necessary.
 *
 * @tparam Derived The implementing engine (ClassicalEngine or QuantumEngine).
 */
template <typename Derived> class BaseKMeansEngine : public KMeansEngine {
  protected:
    float* m_d_samples = nullptr; ///< Sample data in GPU memory
    float* m_d_centers = nullptr; ///< Current centroids in GPU memory
    int* m_d_labels = nullptr;    ///< Assigned cluster labels for each sample on GPU
    float* m_d_newSums = nullptr; ///< Intermediate centroid sums for the Update step
    int* m_d_counts = nullptr;    ///< Point counts per cluster for the Update step
    int* m_d_changed = nullptr;   ///< Flag indicating if any label changed this iteration
    size_t m_maxPoints = 0;       ///< Tracked maximum capacity of sample buffer
    int m_maxK = 0;               ///< Tracked maximum capacity for clusters

  public:
    BaseKMeansEngine() = default;
    ~BaseKMeansEngine() override;

    /**
     * @brief High-level entry point for CPU-resident data.
     * Uploads data to GPU and triggers the internal loop.
     */
    [[nodiscard]] std::vector<FeatureVector> run(const cv::Mat& samples, std::span<const FeatureVector> initialCenters,
                                                 int k, int maxIterations) override;

    /**
     * @brief High-level entry point for GPU-resident data.
     * Operates directly on VRAM to avoid transfer overhead.
     */
    [[nodiscard]] std::vector<FeatureVector> runOnDevice(float* d_samples_ext, int numPoints,
                                                         std::span<const FeatureVector> initialCenters, int k,
                                                         int maxIterations) override;

  protected:
    /** @brief Lazily allocates or resizes GPU buffers to accommodate requested workload. */
    void ensureBuffers(int numPoints, int k);

    /**
     * @brief The core K-Means iteration logic.
     *
     * Orchestrates the two-step Lloyd's algorithm:
     * 1. Assignment Step: Launches the implementation-specific kernel (Classical/Quantum).
     * 2. Update Step: Computes new centroid positions based on current assignments.
     */
    [[nodiscard]] std::vector<FeatureVector> runInternal(float* d_samples, int numPoints,
                                                         std::span<const FeatureVector> initialCenters, int k,
                                                         int maxIterations);

    /** @brief Triggers the generic CUDA kernel for the Update step of K-Means. */
    void baseUpdateKernel(float* d_samples, int numPoints, int k, int* d_labels, float* d_newSums, int* d_counts,
                          int threadsPerBlock, int blocksPerGrid, size_t sharedSize) const;
};

} // namespace kmeans::clustering
