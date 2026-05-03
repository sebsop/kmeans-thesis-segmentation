#pragma once

#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

template <typename Derived>
class BaseKMeansEngine : public KMeansEngine {
  protected:
    float* m_d_samples = nullptr;
    float* m_d_centers = nullptr;
    int* m_d_labels = nullptr;
    float* m_d_newSums = nullptr;
    int* m_d_counts = nullptr;
    int* m_d_changed = nullptr;
    size_t m_maxPoints = 0;
    int m_maxK = 0;

  public:
    BaseKMeansEngine() = default;
    ~BaseKMeansEngine() override;

    [[nodiscard]] std::vector<FeatureVector>
    run(const cv::Mat& samples, std::span<const FeatureVector> initialCenters, int k,
        int maxIterations) override;

    [[nodiscard]] std::vector<FeatureVector>
    runOnDevice(float* d_samples_ext, int numPoints,
                std::span<const FeatureVector> initialCenters, int k,
                int maxIterations) override;

  protected:
    void ensureBuffers(int numPoints, int k);

    [[nodiscard]] std::vector<FeatureVector>
    runInternal(float* d_samp, int numPoints,
                std::span<const FeatureVector> initialCenters, int k, int maxIterations);

    void baseUpdateKernel(float* d_samp, int numPoints, int k, int* d_lab, float* d_nSums, int* d_cnts,
                          int threadsPerBlock, int blocksPerGrid, size_t sharedSize) const;
};

} // namespace kmeans::clustering
