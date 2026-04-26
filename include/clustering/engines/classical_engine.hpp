#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "clustering/engines/kmeans_engine.hpp"

namespace kmeans::clustering {

class ClassicalEngine final : public KMeansEngine {
  private:
    float* d_samples = nullptr;
    float* d_centers = nullptr;
    int*   d_labels  = nullptr;
    float* d_newSums = nullptr;
    int*   d_counts  = nullptr;
    int*   d_changed = nullptr;
    size_t m_maxPoints = 0;
    int    m_maxK      = 0;

  public:
    ClassicalEngine() = default;
    ~ClassicalEngine();

    [[nodiscard]] std::vector<cv::Vec<float, 5>>
    run(const cv::Mat& samples, const std::vector<cv::Vec<float, 5>>& initialCenters, int k, int maxIterations) override;

    [[nodiscard]] std::vector<cv::Vec<float, 5>> runOnDevice(float* d_samples_ext, int numPoints,
                                                             const std::vector<cv::Vec<float, 5>>& initialCenters,
                                                             int k, int maxIterations) override;

  private:
    void ensureBuffers(int numPoints, int k);
    [[nodiscard]] std::vector<cv::Vec<float, 5>> runInternal(float* d_samp, int numPoints,
                                                              const std::vector<cv::Vec<float, 5>>& initialCenters,
                                                              int k, int maxIterations);
};

} // namespace kmeans::clustering