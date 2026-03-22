#include "clustering/clustering_manager.hpp"
#include "clustering/clustering_factory.hpp"
#include "common/enums.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"
#include "opencv2/core.hpp"

namespace kmeans {
namespace clustering {

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
        if (!m_clusteringEngine) {
            m_clusteringEngine = ClusteringFactory::createEngine(m_config);
        }
        m_prevConfig = m_config;
    }

    cv::Mat ClusteringManager::segmentFrame(const cv::Mat& frame) {
        std::vector<cv::Vec<float, 5>> centers = computeCenters(frame);

        if (!m_cudaContext || m_cudaContext->getWidth() != frame.cols || m_cudaContext->getK() != m_config.k) {
            m_cudaContext = std::make_unique<CudaAssignmentContext>(frame.cols, frame.rows, m_config.k);
        }

        cv::Mat result(frame.rows, frame.cols, CV_8UC3);

        m_cudaContext->run(
            frame,
            centers,
            result
        );
        
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

        cv::Mat samples = m_dataPreprocessor->prepare(frame);

        std::vector<cv::Vec<float, 5>> initialCenters;
        if (m_hasPrevious && m_previousCenters.size() == m_config.k) {
            initialCenters = m_previousCenters;
        } else {
            initialCenters = m_initializer->initialize(samples, m_config.k);
        }

        std::vector<cv::Vec<float, 5>> finalCenters = m_clusteringEngine->run(samples, initialCenters, m_config.k);

        m_previousCenters = finalCenters;
        m_hasPrevious = true;

        return m_previousCenters;
    }

}
}