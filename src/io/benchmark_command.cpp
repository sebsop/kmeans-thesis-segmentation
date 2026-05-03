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

RunBenchmarkCommand::RunBenchmarkCommand(const cv::Mat& frame, const common::SegmentationConfig& config)
    : m_frame(frame.clone()), m_config(config) {
    m_config.maxIterations = constants::clustering::BENCHMARK_MAX_ITERATIONS; // Guarantee true convergence
}

void RunBenchmarkCommand::execute() {
    cv::Mat benchFrame = m_frame;
    common::SegmentationConfig benchConfig = m_config;

    m_future = std::async(std::launch::async, [benchFrame, benchConfig]() {
        BenchmarkComparisonResult result;
        result.originalFrame = benchFrame;

        cv::Mat smallFrame;
        cv::resize(benchFrame, smallFrame, cv::Size(constants::video::PROCESS_WIDTH, constants::video::PROCESS_HEIGHT));

        std::vector<FeatureVector> sharedCenters;
        {
            clustering::ClusteringManager initMgr;
            initMgr.getConfig() = benchConfig;
            sharedCenters = initMgr.generateInitialCenters(smallFrame);
        }

        auto runEngine = [&](common::AlgorithmType algo, cv::Mat& outSeg,
                             clustering::metrics::BenchmarkResults& outMetrics,
                             std::vector<FeatureVector>& outCenters) {
            clustering::ClusteringManager mgr;
            common::SegmentationConfig cfg = benchConfig;
            cfg.algorithm = algo;
            mgr.getConfig() = cfg;
            mgr.setInitialCenters(sharedCenters);

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat segmented = mgr.segmentFrame(smallFrame);
            auto end = std::chrono::high_resolution_clock::now();
            float execMs = std::chrono::duration<float, std::milli>(end - start).count();

            outCenters = mgr.getCenters();
            int iterations = mgr.getEngine() ? mgr.getEngine()->getLastIterations() : 0;
            cv::resize(segmented, outSeg, benchFrame.size(), 0, 0, constants::viz::RESIZE_ALGO);

            int n = smallFrame.rows * smallFrame.cols;
            cv::Mat samples(n, constants::clustering::FEATURE_DIMS, CV_32F);
            float colorScale = constants::video::COLOR_SCALE;
            float spatialScale = constants::video::SPATIAL_SCALE;

            float invCols = 1.0f / static_cast<float>(smallFrame.cols);
            float invRows = 1.0f / static_cast<float>(smallFrame.rows);

            std::vector<int> pixel_indices(n);
            std::iota(pixel_indices.begin(), pixel_indices.end(), 0);
            std::for_each(pixel_indices.begin(), pixel_indices.end(), [&](int idx) {
                int y = idx / smallFrame.cols;
                int x = idx % smallFrame.cols;
                cv::Vec3b px = smallFrame.at<cv::Vec3b>(y, x);
                auto* ptr = samples.ptr<float>(idx);
                ptr[0] = static_cast<float>(px[0]) * colorScale;
                ptr[1] = static_cast<float>(px[1]) * colorScale;
                ptr[2] = static_cast<float>(px[2]) * colorScale;
                // Match preprocessor logic: (x / cols) * spatialScale
                ptr[3] = (static_cast<float>(x) * invCols) * spatialScale;
                ptr[4] = (static_cast<float>(y) * invRows) * spatialScale;
            });
            outMetrics = clustering::metrics::computeAllMetrics(samples, outCenters, iterations, execMs);
        };

        runEngine(common::AlgorithmType::KMEANS_REGULAR, result.classicalSegmented, result.classicalMetrics,
                  result.classicalCenters);
        runEngine(common::AlgorithmType::KMEANS_QUANTUM, result.quantumSegmented, result.quantumMetrics,
                  result.quantumCenters);

        return result;
    });
}

} // namespace kmeans::io
