#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/kmeans_engine.hpp"

namespace kmeans::clustering {

/**
 * @brief Base implementation for KMeans engines using the Template Method Pattern.
 * 
 * Provides shared memory management, synchronization, and the iterative execution loop.
 * Subclasses only need to implement the specific kernel launching logic.
 */
class BaseKMeansEngine : public KMeansEngine {
  protected:
    float* d_samples = nullptr;
    float* d_centers = nullptr;
    int* d_labels = nullptr;
    float* d_newSums = nullptr;
    int* d_counts = nullptr;
    int* d_changed = nullptr;
    size_t m_maxPoints = 0;
    int m_maxK = 0;

  public:
    BaseKMeansEngine() = default;
    virtual ~BaseKMeansEngine();

    [[nodiscard]] std::vector<cv::Vec<float, 5>> run(const cv::Mat& samples,
                                                     const std::vector<cv::Vec<float, 5>>& initialCenters, int k,
                                                     int maxIterations) override;

    [[nodiscard]] std::vector<cv::Vec<float, 5>> runOnDevice(float* d_samples_ext, int numPoints,
                                                             const std::vector<cv::Vec<float, 5>>& initialCenters,
                                                             int k, int maxIterations) override;

  protected:
    void ensureBuffers(int numPoints, int k);
    
    [[nodiscard]] std::vector<cv::Vec<float, 5>> runInternal(float* d_samp, int numPoints,
                                                             const std::vector<cv::Vec<float, 5>>& initialCenters,
                                                             int k, int maxIterations);

    /**
     * @brief Hook for subclasses to perform any setup before the iterative loop begins.
     * @param initialCenters The initial cluster centers.
     * @param samples Host-side samples matrix (empty if runOnDevice is used).
     */
    virtual void preRunSetup(const std::vector<cv::Vec<float, 5>>& initialCenters, const cv::Mat& samples) {}

    /**
     * @brief Hook to launch the assignment kernel.
     */
    virtual void launchAssignKernel(float* d_samp, int numPoints, float* d_cent, int k,
                                    int* d_lab, int* d_chg, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) = 0;

    /**
     * @brief Hook to launch the update kernel.
     */
    virtual void launchUpdateKernel(float* d_samp, int numPoints, int k,
                                    int* d_lab, float* d_nSums, int* d_cnts, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) = 0;
};

} // namespace kmeans::clustering
