#include "clustering/clustering_manager.hpp"

#include <opencv2/core.hpp>

#include "backend/cuda_assignment_context.hpp"
#include "clustering/clustering_factory.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "clustering/initializers/initializer.hpp"
#include "clustering/preprocessor/strided_data_preprocessor.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"
#include "common/enums.hpp"

namespace kmeans::clustering {

struct ClusteringManager::Impl {
    std::vector<FeatureVector> m_previousCenters;
    bool m_hasPrevious = false;
    int m_frameCount = 0;

    std::unique_ptr<backend::CudaAssignmentContext> m_cudaContext;
    common::SegmentationConfig m_config;

    std::unique_ptr<StridedDataPreprocessor> m_dataPreprocessor;

    std::unique_ptr<Initializer> m_initializer;
    std::unique_ptr<KMeansEngine> m_clusteringEngine;

    common::SegmentationConfig m_prevConfig;
    std::vector<FeatureVector> m_centers;
};

ClusteringManager::ClusteringManager() : m_impl(std::make_unique<Impl>()) {
    updateStategyImplementations();
}

ClusteringManager::~ClusteringManager() = default;

common::SegmentationConfig& ClusteringManager::getConfig() noexcept { return m_impl->m_config; }
const common::SegmentationConfig& ClusteringManager::getConfig() const noexcept { return m_impl->m_config; }

const std::vector<FeatureVector>& ClusteringManager::getCenters() const noexcept {
    return m_impl->m_centers;
}

KMeansEngine* ClusteringManager::getEngine() const noexcept { return m_impl->m_clusteringEngine.get(); }

void ClusteringManager::resetCenters() {
    m_impl->m_hasPrevious = false;
    if (m_impl->m_dataPreprocessor) {
        m_impl->m_dataPreprocessor->reset();
    }
}

void ClusteringManager::updateStategyImplementations() {
    if (!m_impl->m_dataPreprocessor) {
        m_impl->m_dataPreprocessor = std::make_unique<StridedDataPreprocessor>();
    }
    if (!m_impl->m_initializer || m_impl->m_config.init != m_impl->m_prevConfig.init) {
        m_impl->m_initializer = ClusteringFactory::createInitializer(m_impl->m_config);
    }
    if (!m_impl->m_clusteringEngine || m_impl->m_config.algorithm != m_impl->m_prevConfig.algorithm) {
        m_impl->m_clusteringEngine = ClusteringFactory::createEngine(m_impl->m_config);
    }
    m_impl->m_prevConfig = m_impl->m_config;
}

cv::Mat ClusteringManager::segmentFrame(const cv::Mat& frame) {
    std::vector<FeatureVector> centers = computeCenters(frame);

    if (!m_impl->m_cudaContext || m_impl->m_cudaContext->getWidth() != frame.cols || m_impl->m_cudaContext->getK() != m_impl->m_config.k) {
        m_impl->m_cudaContext = std::make_unique<backend::CudaAssignmentContext>(frame.cols, frame.rows, m_impl->m_config.k);
    }

    cv::Mat result(frame.rows, frame.cols, CV_8UC3);

    m_impl->m_cudaContext->run(frame, centers, result);

    m_impl->m_centers = centers;
    return result;
}

std::vector<FeatureVector> ClusteringManager::computeCenters(const cv::Mat& frame) {
    CV_Assert(!frame.empty() && "Input frame is empty in ClusteringManager::computeCenters");
    CV_Assert(m_impl->m_config.k > 0 && "Number of clusters 'k' must be greater than 0");

    updateStategyImplementations(); // Allow hot-swapping if config changed

    bool shouldUpdate = (m_impl->m_frameCount % m_impl->m_config.learningInterval == 0) || !m_impl->m_hasPrevious;
    m_impl->m_frameCount++;

    if (!shouldUpdate && m_impl->m_hasPrevious) {
        return m_impl->m_previousCenters;
    }

    std::vector<FeatureVector> initialCenters;
    std::vector<FeatureVector> finalCenters;

    if (auto* sdp = dynamic_cast<StridedDataPreprocessor*>(m_impl->m_dataPreprocessor.get())) {
        int numPoints = 0;
        float* d_samples = sdp->prepareDevice(frame, m_impl->m_config.stride, numPoints);

        if (m_impl->m_hasPrevious && static_cast<int>(m_impl->m_previousCenters.size()) == m_impl->m_config.k) {
            initialCenters = m_impl->m_previousCenters;
        } else {
            cv::Mat cpuSamples = sdp->download();
            initialCenters = m_impl->m_initializer->initialize(cpuSamples, m_impl->m_config.k);
        }

        finalCenters =
            m_impl->m_clusteringEngine->runOnDevice(d_samples, numPoints, initialCenters, m_impl->m_config.k, m_impl->m_config.maxIterations);
    } else {
        cv::Mat samples = m_impl->m_dataPreprocessor->prepare(frame);

        if (m_impl->m_hasPrevious && static_cast<int>(m_impl->m_previousCenters.size()) == m_impl->m_config.k) {
            initialCenters = m_impl->m_previousCenters;
        } else {
            initialCenters = m_impl->m_initializer->initialize(samples, m_impl->m_config.k);
        }

        finalCenters = m_impl->m_clusteringEngine->run(samples, initialCenters, m_impl->m_config.k, m_impl->m_config.maxIterations);
    }

    m_impl->m_previousCenters = finalCenters;
    m_impl->m_hasPrevious = true;

    return m_impl->m_previousCenters;
}

std::vector<FeatureVector> ClusteringManager::generateInitialCenters(const cv::Mat& frame) {
    updateStategyImplementations();

    cv::Mat cpuSamples;
    if (auto* sdp = dynamic_cast<StridedDataPreprocessor*>(m_impl->m_dataPreprocessor.get())) {
        int dummyPoints = 0;
        sdp->prepareDevice(frame, m_impl->m_config.stride, dummyPoints);
        cpuSamples = sdp->download();
    } else {
        cpuSamples = m_impl->m_dataPreprocessor->prepare(frame);
    }

    return m_impl->m_initializer->initialize(cpuSamples, m_impl->m_config.k);
}

void ClusteringManager::setInitialCenters(std::span<const FeatureVector> centers) {
    m_impl->m_previousCenters = std::vector<FeatureVector>(centers.begin(), centers.end());
    m_impl->m_hasPrevious = true;
}

} // namespace kmeans::clustering
