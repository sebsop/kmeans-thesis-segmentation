#pragma once

#include <memory>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "common/constants.hpp"

namespace kmeans::backend {
class CudaAssignmentContext;
}

namespace kmeans::clustering {
class KMeansEngine;

class ClusteringManager {
  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

  public:
    /** @brief Basic Exception Safety Guarantee */
    ClusteringManager();
    /** @brief No-throw Exception Safety Guarantee */
    ~ClusteringManager();

    [[nodiscard]] common::SegmentationConfig& getConfig() noexcept;
    [[nodiscard]] const common::SegmentationConfig& getConfig() const noexcept;

    [[nodiscard]] const std::vector<FeatureVector>& getCenters() const noexcept;

    [[nodiscard]] KMeansEngine* getEngine() const noexcept;

    /** @brief Strong Exception Safety Guarantee */
    void updateStategyImplementations();
    /** @brief No-throw Exception Safety Guarantee */
    void resetCenters();

    /** @brief Strong Exception Safety Guarantee */
    void setInitialCenters(std::span<const FeatureVector> centers);

    /** @brief Basic Exception Safety Guarantee */
    std::vector<FeatureVector> generateInitialCenters(const cv::Mat& frame);

    /** @brief Strong Exception Safety Guarantee */
    [[nodiscard]] cv::Mat segmentFrame(const cv::Mat& frame);
    /** @brief Basic Exception Safety Guarantee */
    [[nodiscard]] std::vector<cv::Vec<float, constants::FEATURE_DIMS>> computeCenters(const cv::Mat& frame);
};

} // namespace kmeans::clustering