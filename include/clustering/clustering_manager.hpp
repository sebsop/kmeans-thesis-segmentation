#pragma once

#include <memory>
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
    ClusteringManager();
    ~ClusteringManager();

    [[nodiscard]] common::SegmentationConfig& getConfig() noexcept;
    [[nodiscard]] const common::SegmentationConfig& getConfig() const noexcept;

    [[nodiscard]] const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& getCenters() const noexcept;

    [[nodiscard]] KMeansEngine* getEngine() const noexcept;

    void updateStategyImplementations();
    void resetCenters();

    void setInitialCenters(const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& centers);

    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> generateInitialCenters(const cv::Mat& frame);

    [[nodiscard]] cv::Mat segmentFrame(const cv::Mat& frame);
    [[nodiscard]] std::vector<cv::Vec<float, constants::FEATURE_DIMS>> computeCenters(const cv::Mat& frame);
};

} // namespace kmeans::clustering