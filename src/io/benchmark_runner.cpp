#include "io/benchmark_runner.hpp"

#include "clustering/clustering_manager.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"
#include <opencv2/imgproc.hpp>

namespace kmeans::io {

void BenchmarkRunner::requestCapture() {
    m_state = BenchmarkState::CAPTURING;
    m_statusText = "Requesting frame from camera thread...";
}

void BenchmarkRunner::requestRecompute() {
    m_state = BenchmarkState::RECOMPUTING;
}

void BenchmarkRunner::reset() {
    m_state = BenchmarkState::IDLE;
    m_results.reset();
}

void BenchmarkRunner::startComputing(const cv::Mat& currentFrame, const common::SegmentationConfig& config) {
    bool isRecomputing = (m_state == BenchmarkState::RECOMPUTING);
    m_state = BenchmarkState::COMPUTING;
    m_statusText = "Extracting frame and running dual-engine comparison...";
    
    cv::Mat benchFrame;
    if (isRecomputing && m_results.has_value()) {
        benchFrame = m_results->originalFrame.clone();
    } else {
        benchFrame = currentFrame.clone();
    }
    
    common::SegmentationConfig benchConfig = config;
    benchConfig.maxIterations = 1000; // Let benchmark run until true convergence

    m_future = std::async(std::launch::async, [benchFrame, benchConfig]() {
        BenchmarkComparisonResult result;
        result.originalFrame = benchFrame.clone();

        cv::Mat smallFrame;
        cv::resize(benchFrame, smallFrame,
                   cv::Size(constants::PROCESS_WIDTH, constants::PROCESS_HEIGHT));

        // Generate shared initial centers to guarantee a fair comparison
        std::vector<cv::Vec<float, 5>> sharedCenters;
        {
            clustering::ClusteringManager initMgr;
            initMgr.getConfig() = benchConfig;
            sharedCenters = initMgr.generateInitialCenters(smallFrame);
        }

        auto runEngine = [&](common::AlgorithmType algo, cv::Mat& outSeg,
                             clustering::metrics::BenchmarkResults& outMetrics,
                             std::vector<cv::Vec<float, 5>>& outCenters) {
            clustering::ClusteringManager mgr;
            common::SegmentationConfig cfg = benchConfig;
            cfg.algorithm = algo;
            mgr.getConfig() = cfg;
            mgr.setInitialCenters(sharedCenters); // Force both engines to use the exact same starting points

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat segmented = mgr.segmentFrame(smallFrame);
            auto end = std::chrono::high_resolution_clock::now();
            float execMs = std::chrono::duration<float, std::milli>(end - start).count();

            outCenters = mgr.getCenters();
            int iterations = mgr.getEngine() ? mgr.getEngine()->getLastIterations() : 0;
            cv::resize(segmented, outSeg, benchFrame.size(), 0, 0, cv::INTER_NEAREST);

            int n = smallFrame.rows * smallFrame.cols;
            cv::Mat samples(n, 5, CV_32F);
            float colorScale = constants::COLOR_SCALE;
            float spatialScale = constants::SPATIAL_SCALE;
            for (int y = 0; y < smallFrame.rows; ++y) {
                for (int x = 0; x < smallFrame.cols; ++x) {
                    cv::Vec3b px = smallFrame.at<cv::Vec3b>(y, x);
                    int idx = (y * smallFrame.cols) + x;
                    auto* ptr = samples.ptr<float>(idx);
                    ptr[0] = static_cast<float>(px[0]) * colorScale;
                    ptr[1] = static_cast<float>(px[1]) * colorScale;
                    ptr[2] = static_cast<float>(px[2]) * colorScale;
                    ptr[3] = static_cast<float>(x) * spatialScale;
                    ptr[4] = static_cast<float>(y) * spatialScale;
                }
            }
            outMetrics = clustering::metrics::computeAllMetrics(samples, outCenters, iterations, execMs);
        };

        runEngine(common::AlgorithmType::KMEANS_REGULAR, result.classicalSegmented, result.classicalMetrics, result.classicalCenters);
        runEngine(common::AlgorithmType::KMEANS_QUANTUM, result.quantumSegmented, result.quantumMetrics, result.quantumCenters);

        return result;
    });
}

void BenchmarkRunner::poll() {
    if (m_state == BenchmarkState::COMPUTING) {
        if (m_future.valid() && m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            m_results = m_future.get();
            m_state = BenchmarkState::DONE;
            m_statusText = "Benchmark Complete.";
        }
    }
}

} // namespace kmeans::io
