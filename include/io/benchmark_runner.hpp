#pragma once

#include <chrono>
#include <future>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "clustering/metrics.hpp"
#include "common/config.hpp"

namespace kmeans::io {

enum class BenchmarkState { IDLE, CAPTURING, RECOMPUTING, COMPUTING, DONE };

struct BenchmarkComparisonResult {
    cv::Mat originalFrame;
    cv::Mat classicalSegmented;
    cv::Mat quantumSegmented;
    clustering::metrics::BenchmarkResults classicalMetrics{};
    clustering::metrics::BenchmarkResults quantumMetrics{};
    std::vector<cv::Vec<float, 5>> classicalCenters;
    std::vector<cv::Vec<float, 5>> quantumCenters;
};

/**
 * @brief Manages the execution and state of side-by-side clustering benchmarks.
 * 
 * Uses std::async to offload the heavy benchmark computations from the UI and camera threads.
 */
class BenchmarkRunner {
  private:
    BenchmarkState m_state = BenchmarkState::IDLE;
    std::future<BenchmarkComparisonResult> m_future;
    std::optional<BenchmarkComparisonResult> m_results;
    std::string m_statusText;

  public:
    BenchmarkRunner() = default;
    ~BenchmarkRunner() = default;

    [[nodiscard]] BenchmarkState getState() const noexcept { return m_state; }
    [[nodiscard]] const std::string& getStatusText() const noexcept { return m_statusText; }
    [[nodiscard]] std::optional<BenchmarkComparisonResult>& getResults() noexcept { return m_results; }

    void requestCapture();
    void requestRecompute();
    void reset();

    /**
     * @brief Triggers the actual background execution.
     */
    void startComputing(const cv::Mat& currentFrame, const common::SegmentationConfig& config);

    /**
     * @brief Polls the future to see if computation is done.
     */
    void poll();
};

} // namespace kmeans::io
