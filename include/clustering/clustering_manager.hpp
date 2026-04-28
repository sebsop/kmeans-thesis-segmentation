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
class Initializer;
class StridedDataPreprocessor;

/**
 * @brief Strategy Context and Manager class.
 *
 * This class acts as the Context in the Strategy Design Pattern. It owns the
 * strategy objects created via the ClusteringFactory and coordinates their
 * execution to run the K-Means pipeline.
 */
class ClusteringManager {
  private:
    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> m_previousCenters;
    bool m_hasPrevious = false;
    int m_frameCount = 0;

    std::unique_ptr<backend::CudaAssignmentContext> m_cudaContext;
    common::SegmentationConfig m_config;

    // Concrete Preprocessor (No longer a strategy)
    std::unique_ptr<StridedDataPreprocessor> m_dataPreprocessor;

    // Active Strategy Implementations
    std::unique_ptr<Initializer> m_initializer;
    std::unique_ptr<KMeansEngine> m_clusteringEngine;

    common::SegmentationConfig m_prevConfig;
    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> m_centers;

  public:
    ClusteringManager();
    ~ClusteringManager();

    [[nodiscard]] common::SegmentationConfig& getConfig() noexcept { return m_config; }
    [[nodiscard]] const common::SegmentationConfig& getConfig() const noexcept { return m_config; }

    [[nodiscard]] const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& getCenters() const noexcept {
        return m_centers;
    }

    [[nodiscard]] KMeansEngine* getEngine() const noexcept { return m_clusteringEngine.get(); }

    /** @brief Reconstructs strategy instances using the factory if config changed. */
    void updateStategyImplementations();

    /** @brief Resets the clustering state, clearing previous centers and caches. */
    void resetCenters();

    /** @brief Forces the engine to use a specific set of initial centroids (bypasses initializer). */
    void setInitialCenters(const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& centers) {
        m_previousCenters = centers;
        m_hasPrevious = true;
    }

    /** @brief Generates initial centroids using the currently configured initializer without running K-Means. */
    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> generateInitialCenters(const cv::Mat& frame);

    /** @brief Segments a single frame using the current clustering configuration. */
    [[nodiscard]] cv::Mat segmentFrame(const cv::Mat& frame);
    [[nodiscard]] std::vector<cv::Vec<float, constants::FEATURE_DIMS>> computeCenters(const cv::Mat& frame);
};

} // namespace kmeans::clustering