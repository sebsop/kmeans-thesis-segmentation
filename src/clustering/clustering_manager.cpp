#include "clustering/clustering_manager.hpp"

#include <opencv2/core.hpp>

#include "clustering/clustering_factory.hpp"
#include "clustering/preprocessors/full_data_preprocessor.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"
#include "common/enums.hpp"

namespace kmeans::clustering {

ClusteringManager::ClusteringManager() {
    updateStategyImplementations();
}

void ClusteringManager::updateStategyImplementations() {
    if (!m_dataPreprocessor || m_config.strategy != m_prevConfig.strategy) {
        m_dataPreprocessor = ClusteringFactory::createDataPreprocessor(m_config);
    }
    if (!m_initializer || m_config.init != m_prevConfig.init) {
        m_initializer = ClusteringFactory::createInitializer(m_config);
    }
    if (!m_clusteringEngine || m_config.algorithm != m_prevConfig.algorithm) {
        m_clusteringEngine = ClusteringFactory::createEngine(m_config);
    }
    m_prevConfig = m_config;
}

cv::Mat ClusteringManager::segmentFrame(const cv::Mat& frame) {
    std::vector<cv::Vec<float, 5>> centers = computeCenters(frame);

    if (!m_cudaContext || m_cudaContext->getWidth() != frame.cols || m_cudaContext->getK() != m_config.k) {
        m_cudaContext = std::make_unique<backend::CudaAssignmentContext>(frame.cols, frame.rows, m_config.k);
    }

    cv::Mat result(frame.rows, frame.cols, CV_8UC3);

    m_cudaContext->run(frame, centers, result);

    m_centers = centers;
    return result;
}

std::vector<cv::Vec<float, 5>> ClusteringManager::computeCenters(const cv::Mat& frame) {
    CV_Assert(!frame.empty() && "Input frame is empty in ClusteringManager::computeCenters");
    CV_Assert(m_config.k > 0 && "Number of clusters 'k' must be greater than 0");

    updateStategyImplementations(); // Allow hot-swapping if config changed

    bool shouldUpdate = (m_frameCount % m_config.learningInterval == 0) || !m_hasPrevious;
    m_frameCount++;

    if (!shouldUpdate && m_hasPrevious) {
        return m_previousCenters;
    }

    std::vector<cv::Vec<float, 5>> initialCenters;
    std::vector<cv::Vec<float, 5>> finalCenters;

    // GPU-direct path: FullDataPreprocessor can hand off device-side samples directly,
    // eliminating the D2H+H2D round-trip (saves ~6MB PCIe traffic per processed frame).
    if (auto* fdp = dynamic_cast<FullDataPreprocessor*>(m_dataPreprocessor.get())) {
        int numPoints = 0;
        float* d_samples = fdp->prepareDevice(frame, numPoints);

        if (m_hasPrevious && static_cast<int>(m_previousCenters.size()) == m_config.k) {
            initialCenters = m_previousCenters;
        } else {
            // Initializer still needs CPU samples for K-Means++ (tiny, one-time cost)
            cv::Mat cpuSamples = fdp->download();
            initialCenters = m_initializer->initialize(cpuSamples, m_config.k);
        }

        finalCenters = m_clusteringEngine->runOnDevice(d_samples, numPoints, initialCenters, m_config.k);
    } else {
        // CPU path (RCC coreset preprocessor or future preprocessors)
        cv::Mat samples = m_dataPreprocessor->prepare(frame);

        if (m_hasPrevious && static_cast<int>(m_previousCenters.size()) == m_config.k) {
            initialCenters = m_previousCenters;
        } else {
            initialCenters = m_initializer->initialize(samples, m_config.k);
        }

        finalCenters = m_clusteringEngine->run(samples, initialCenters, m_config.k);
    }

    m_previousCenters = finalCenters;
    m_hasPrevious = true;

    return m_previousCenters;
}

} // namespace kmeans::clustering