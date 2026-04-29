#include "io/benchmark_command.hpp"
#include "io/benchmark_runner.hpp"

#include <chrono>
#include <future>
#include <numeric>
#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "clustering/clustering_manager.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::io {

RunBenchmarkCommand::RunBenchmarkCommand(const cv::Mat& frame, const common::SegmentationConfig& config)
    : m_frame(frame.clone()), m_config(config) {
    m_config.maxIterations = constants::BENCHMARK_MAX_ITERATIONS; // Guarantee true convergence
}

void RunBenchmarkCommand::execute() {
    cv::Mat benchFrame = m_frame;
    common::SegmentationConfig benchConfig = m_config;

    m_future = std::async(std::launch::async, [benchFrame, benchConfig]() {
        BenchmarkComparisonResult result;
        result.originalFrame = benchFrame;

        cv::Mat smallFrame;
        cv::resize(benchFrame, smallFrame, cv::Size(constants::PROCESS_WIDTH, constants::PROCESS_HEIGHT));

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
            cv::resize(segmented, outSeg, benchFrame.size(), 0, 0, constants::VIZ_RESIZE_ALGO);

            int n = smallFrame.rows * smallFrame.cols;
            cv::Mat samples(n, 5, CV_32F);
            float colorScale = constants::COLOR_SCALE;
            float spatialScale = constants::SPATIAL_SCALE;
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
                ptr[3] = static_cast<float>(x) * spatialScale;
                ptr[4] = static_cast<float>(y) * spatialScale;
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
