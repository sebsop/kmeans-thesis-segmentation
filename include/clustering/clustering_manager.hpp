#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include "backend/cuda_assignment_context.hpp"
#include "common/config.hpp"
#include "clustering/preprocessors/data_preprocessor.hpp"
#include "clustering/initializers/initializer.hpp"
#include "clustering/engines/kmeans_engine.hpp"

namespace kmeans {
namespace clustering {

    class ClusteringManager {
    private:
        std::vector<cv::Vec<float, 5>> m_previousCenters;
        bool m_hasPrevious = false;
        int m_frameCount = 0;
        std::unique_ptr<CudaAssignmentContext> m_cudaContext;
        SegmentationConfig m_config;

        std::unique_ptr<DataPreprocessor> m_dataPreprocessor;
        std::unique_ptr<Initializer> m_initializer;
        std::unique_ptr<KMeansEngine> m_clusteringEngine;

        SegmentationConfig m_prevConfig;
        std::vector<cv::Vec<float, 5>> m_centers;

    public:
        ClusteringManager();
        ~ClusteringManager() = default;

        SegmentationConfig& getConfig() { return m_config; }
        const SegmentationConfig& getConfig() const { return m_config; }
        
        const std::vector<cv::Vec<float, 5>>& getCenters() const { return m_centers; }
        // Factory setup function to inject strategies based on config
        void updateStategyImplementations();

        cv::Mat segmentFrame(const cv::Mat& frame);
        std::vector<cv::Vec<float, 5>> computeCenters(const cv::Mat& frame);
    };

}
}