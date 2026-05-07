/**
 * @file clustering_manager.cpp
 * @brief Implementation of the central clustering orchestration logic.
 */

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

/**
 * @struct ClusteringManager::Impl
 * @brief Pointer to IMPLementation (PIMPL) for the ClusteringManager.
 *
 * This structure encapsulates the concrete strategy objects (Preprocessor,
 * Initializer, Engine) and the temporal state (frame counts, previous centers)
 * required to maintain stability across video frames.
 */
struct ClusteringManager::Impl {
    std::vector<FeatureVector> m_previousCenters; ///< Centroids from the last learning step
    bool m_hasPrevious = false;                   ///< True if we have valid history
    int m_frameCount = 0;                         ///< Counter for temporal subsampling

    std::unique_ptr<backend::CudaAssignmentContext> m_cudaContext; ///< Backend for pixel assignment
    common::SegmentationConfig m_config;                           ///< Current user settings

    std::unique_ptr<StridedDataPreprocessor> m_dataPreprocessor; ///< GPU feature extractor

    std::unique_ptr<Initializer> m_initializer;       ///< Current seeding strategy
    std::unique_ptr<KMeansEngine> m_clusteringEngine; ///< Current K-Means backend

    common::SegmentationConfig m_prevConfig; ///< Cache for hot-swap detection
    std::vector<FeatureVector> m_centers;    ///< Current iteration centroids
};

ClusteringManager::ClusteringManager() : m_impl(std::make_unique<Impl>()) {
    updateStategyImplementations();
}

ClusteringManager::~ClusteringManager() = default;

common::SegmentationConfig& ClusteringManager::getConfig() noexcept {
    return m_impl->m_config;
}
const common::SegmentationConfig& ClusteringManager::getConfig() const noexcept {
    return m_impl->m_config;
}

const std::vector<FeatureVector>& ClusteringManager::getCenters() const noexcept {
    return m_impl->m_centers;
}

KMeansEngine* ClusteringManager::getEngine() const noexcept {
    return m_impl->m_clusteringEngine.get();
}

void ClusteringManager::resetCenters() {
    m_impl->m_hasPrevious = false;
    if (m_impl->m_dataPreprocessor) {
        m_impl->m_dataPreprocessor->reset();
    }
}

/**
 * @brief Dynamically updates strategy objects if the configuration has changed.
 *
 * This enables "Hot-Swapping" of the mathematical engines or initialization
 * logic in real-time without restarting the application.
 */
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
    // 1. Determine new cluster centers (may use temporal caching)
    std::vector<FeatureVector> centers = computeCenters(frame);

    // 2. Ensure GPU assignment context matches current frame size
    if (!m_impl->m_cudaContext || m_impl->m_cudaContext->getWidth() != frame.cols ||
        m_impl->m_cudaContext->getK() != m_impl->m_config.k) {
        m_impl->m_cudaContext =
            std::make_unique<backend::CudaAssignmentContext>(frame.cols, frame.rows, m_impl->m_config.k);
    }

    cv::Mat result(frame.rows, frame.cols, CV_8UC3);

    // 3. High-speed GPU assignment
    m_impl->m_cudaContext->run(frame, centers, result);

    m_impl->m_centers = centers;
    return result;
}

/**
 * @brief Orchestrates the K-Means optimization loop.
 *
 * This method handles the logic of whether to perform a full re-clustering
 * based on the 'learningInterval'. If a full run is required, it utilizes
 * the StridedDataPreprocessor for GPU-accelerated feature extraction and
 * executes the selected K-Means engine.
 */
std::vector<FeatureVector> ClusteringManager::computeCenters(const cv::Mat& frame) {
    CV_Assert(!frame.empty());
    CV_Assert(m_impl->m_config.k > 0);

    updateStategyImplementations();

    // Temporal logic: only re-cluster every N frames to save energy and stabilize result
    bool shouldUpdate = (m_impl->m_frameCount % m_impl->m_config.learningInterval == 0) || !m_impl->m_hasPrevious;
    m_impl->m_frameCount++;

    if (!shouldUpdate && m_impl->m_hasPrevious) {
        return m_impl->m_previousCenters;
    }

    std::vector<FeatureVector> initialCenters;
    std::vector<FeatureVector> finalCenters;

    // HIGH PERFORMANCE PATH: Using GPU Preprocessor directly
    if (auto* stridedPreprocessor = dynamic_cast<StridedDataPreprocessor*>(m_impl->m_dataPreprocessor.get())) {
        int numPoints = 0;
        float* d_samples = stridedPreprocessor->prepareDevice(frame, m_impl->m_config.stride, numPoints);

        if (m_impl->m_hasPrevious && static_cast<int>(m_impl->m_previousCenters.size()) == m_impl->m_config.k) {
            initialCenters = m_impl->m_previousCenters;
        } else {
            // Seeding still requires CPU sampling for some strategies
            cv::Mat cpuSamples = stridedPreprocessor->download();
            initialCenters = m_impl->m_initializer->initialize(cpuSamples, m_impl->m_config.k);
        }

        // Run Lloyd's loop directly on GPU features (Zero-Copy)
        finalCenters = m_impl->m_clusteringEngine->runOnDevice(d_samples, numPoints, initialCenters, m_impl->m_config.k,
                                                               m_impl->m_config.maxIterations);
    } else {
        // LEGACY/FALLBACK PATH: Copying features back to CPU
        cv::Mat samples = m_impl->m_dataPreprocessor->prepare(frame);

        if (m_impl->m_hasPrevious && static_cast<int>(m_impl->m_previousCenters.size()) == m_impl->m_config.k) {
            initialCenters = m_impl->m_previousCenters;
        } else {
            initialCenters = m_impl->m_initializer->initialize(samples, m_impl->m_config.k);
        }

        finalCenters = m_impl->m_clusteringEngine->run(samples, initialCenters, m_impl->m_config.k,
                                                       m_impl->m_config.maxIterations);
    }

    m_impl->m_previousCenters = finalCenters;
    m_impl->m_hasPrevious = true;

    return m_impl->m_previousCenters;
}

/**
 * @brief Performs a standalone initialization pass on the given frame.
 *
 * This method runs the preprocessor and initializer to generate a fresh
 * set of centroids. It does NOT update the internal state of the manager;
 * it is primarily used for testing or to preview the seeding logic.
 *
 * @param frame The image to sample for initial centers.
 * @return std::vector<FeatureVector> A new set of seeded centroids.
 */
std::vector<FeatureVector> ClusteringManager::generateInitialCenters(const cv::Mat& frame) {
    updateStategyImplementations();

    cv::Mat cpuSamples;
    if (auto* stridedPreprocessor = dynamic_cast<StridedDataPreprocessor*>(m_impl->m_dataPreprocessor.get())) {
        int preprocessedCount = 0;
        stridedPreprocessor->prepareDevice(frame, m_impl->m_config.stride, preprocessedCount);
        cpuSamples = stridedPreprocessor->download();
    } else {
        cpuSamples = m_impl->m_dataPreprocessor->prepare(frame);
    }

    return m_impl->m_initializer->initialize(cpuSamples, m_impl->m_config.k);
}

/**
 * @brief Manually overrides the internal centroid history.
 *
 * This allows external components (like a UI or a saved-state loader) to
 * inject specific centroids into the pipeline. Subsequent calls to
 * segmentFrame will use these centers as the starting point for optimization.
 *
 * @param centers A span of feature vectors to be used as the new state.
 */
void ClusteringManager::setInitialCenters(std::span<const FeatureVector> centers) {
    m_impl->m_previousCenters = std::vector<FeatureVector>(centers.begin(), centers.end());
    m_impl->m_hasPrevious = true;
}

} // namespace kmeans::clustering
