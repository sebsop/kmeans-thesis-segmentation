/**
 * @file benchmark_command.cpp
 * @brief Implementation of the dual-algorithm comparison task.
 */

#include "io/benchmark_command.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "clustering/clustering_manager.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"
#include "io/benchmark_runner.hpp"

namespace kmeans::io {

/**
 * @brief Constructs a benchmark command for a specific image frame.
 *
 * @param frame The captured image to analyze.
 * @param config User settings (k, stride, etc.) to apply.
 */
RunBenchmarkCommand::RunBenchmarkCommand(const cv::Mat& frame, const common::SegmentationConfig& config)
    : m_frame(frame.clone()), m_config(config) {
    // For benchmarks, we ignore the 'learning interval' and force a full convergence
    // to get a true mathematical baseline.
    m_config.maxIterations = constants::clustering::BENCHMARK_MAX_ITERATIONS;
}

/**
 * @brief Executes the side-by-side comparison in a background thread.
 *
 * This method ensures scientific fairness by:
 * 1. Generating a single set of initial centers (`sharedCenters`).
 * 2. Seeding both the Classical and Quantum engines with these identical centers.
 * 3. Extracting features for the quality metrics pass using logic identical to
 *    the GPU preprocessor.
 */
void RunBenchmarkCommand::execute() {
    cv::Mat benchFrame = m_frame;
    common::SegmentationConfig benchConfig = m_config;

    // Use std::async to run the heavy math without blocking the UI thread.
    m_future = std::async(std::launch::async, [benchFrame, benchConfig]() {
        BenchmarkComparisonResult result;
        result.originalFrame = benchFrame;

        // Resize to a standard processing resolution for consistent benchmarking
        cv::Mat smallFrame;
        cv::resize(benchFrame, smallFrame, cv::Size(constants::video::PROCESS_WIDTH, constants::video::PROCESS_HEIGHT));

        // 1. FAIRNESS STEP: Generate a shared seed
        std::vector<FeatureVector> sharedCenters;
        {
            clustering::ClusteringManager initMgr;
            initMgr.getConfig() = benchConfig;
            sharedCenters = initMgr.generateInitialCenters(smallFrame);
        }

        // Helper to run a specific engine and capture its performance
        auto runEngine = [&](common::AlgorithmType algo, cv::Mat& outSeg,
                             clustering::metrics::BenchmarkResults& outMetrics,
                             std::vector<FeatureVector>& outCenters) {
            clustering::ClusteringManager mgr;
            common::SegmentationConfig cfg = benchConfig;
            cfg.algorithm = algo;
            mgr.getConfig() = cfg;
            mgr.setInitialCenters(sharedCenters); // Seed with shared centers

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat segmented = mgr.segmentFrame(smallFrame);
            auto end = std::chrono::high_resolution_clock::now();
            float execMs = std::chrono::duration<float, std::milli>(end - start).count();

            outCenters = mgr.getCenters();
            int iterations = mgr.getEngine() ? mgr.getEngine()->getLastIterations() : 0;

            // Upscale the result back to original size for UI visualization
            cv::resize(segmented, outSeg, benchFrame.size(), 0, 0, constants::viz::RESIZE_ALGO);

            // 2. METRICS PREPARATION: Extract features for the quality evaluation pass
            int n = smallFrame.rows * smallFrame.cols;
            cv::Mat samples(n, constants::clustering::FEATURE_DIMS, CV_32F);
            float colorScale = constants::video::COLOR_SCALE;
            float spatialScale = constants::video::SPATIAL_WEIGHT;
            float invCols = 1.0f / static_cast<float>(smallFrame.cols);
            float invRows = 1.0f / static_cast<float>(smallFrame.rows);

            for (int idx = 0; idx < n; ++idx) {
                int y = idx / smallFrame.cols;
                int x = idx % smallFrame.cols;
                cv::Vec3b px = smallFrame.at<cv::Vec3b>(y, x);
                auto* ptr = samples.ptr<float>(idx);
                ptr[0] = static_cast<float>(px[0]) * colorScale;
                ptr[1] = static_cast<float>(px[1]) * colorScale;
                ptr[2] = static_cast<float>(px[2]) * colorScale;
                ptr[3] = (static_cast<float>(x) * invCols) * spatialScale;
                ptr[4] = (static_cast<float>(y) * invRows) * spatialScale;
            }

            // Calculate final mathematical quality metrics
            outMetrics = clustering::metrics::computeAllMetrics(samples, outCenters, iterations, execMs);
        };

        // Run both backends sequentially on the worker thread
        runEngine(common::AlgorithmType::KMEANS_REGULAR, result.classicalSegmented, result.classicalMetrics,
                  result.classicalCenters);
        runEngine(common::AlgorithmType::KMEANS_QUANTUM, result.quantumSegmented, result.quantumMetrics,
                  result.quantumCenters);

        return result;
    });
}

} // namespace kmeans::io
