#pragma once

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "backend/cuda_assignment_context.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "clustering/initializers/initializer.hpp"
#include "clustering/preprocessor/strided_data_preprocessor.hpp"
#include "common/config.hpp"

namespace kmeans::clustering {

/**
 * @brief Strategy Context and Manager class.
 *
 * This class acts as the Context in the Strategy Design Pattern. It owns the
 * strategy objects created via the ClusteringFactory and coordinates their
 * execution to run the K-Means pipeline.
 */
class ClusteringManager {
  private:
    std::vector<cv::Vec<float, 5>> m_previousCenters;
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
    std::vector<cv::Vec<float, 5>> m_centers;

  public:
    ClusteringManager();
    ~ClusteringManager() = default;

    [[nodiscard]] common::SegmentationConfig& getConfig() noexcept { return m_config; }
    [[nodiscard]] const common::SegmentationConfig& getConfig() const noexcept { return m_config; }

    [[nodiscard]] const std::vector<cv::Vec<float, 5>>& getCenters() const noexcept { return m_centers; }

    [[nodiscard]] KMeansEngine* getEngine() const noexcept { return m_clusteringEngine.get(); }

    /** @brief Reconstructs strategy instances using the factory if config changed. */
    void updateStategyImplementations();

    /** @brief Resets the clustering state, clearing previous centers and caches. */
    void resetCenters() {
        m_hasPrevious = false;
        if (m_dataPreprocessor) {
            m_dataPreprocessor->reset();
        }
    }

    /** @brief Segments a single frame using the current clustering configuration. */
    [[nodiscard]] cv::Mat segmentFrame(const cv::Mat& frame);
    [[nodiscard]] std::vector<cv::Vec<float, 5>> computeCenters(const cv::Mat& frame);
};

} // namespace kmeans::clustering