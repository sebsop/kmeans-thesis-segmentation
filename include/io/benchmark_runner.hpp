#pragma once

#include <chrono>
#include <future>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "clustering/metrics.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"
#include "io/benchmark_command.hpp"
#include "io/benchmark_observer.hpp"
#include "io/benchmark_result.hpp"

namespace kmeans::io {

enum class BenchmarkState { IDLE, CAPTURING, RECOMPUTING, COMPUTING, DONE };

class BenchmarkRunner {
  private:
    BenchmarkState m_state = BenchmarkState::IDLE;
    std::unique_ptr<IBenchmarkCommand> m_currentCommand;
    std::queue<std::unique_ptr<IBenchmarkCommand>> m_commandQueue;
    std::optional<BenchmarkComparisonResult> m_results;
    std::string m_statusText;
    std::vector<IBenchmarkObserver*> m_observers;

  public:
    BenchmarkRunner() = default;
    ~BenchmarkRunner() = default;

    void addObserver(IBenchmarkObserver* observer);
    void removeObserver(IBenchmarkObserver* observer);
    void notifyObservers(const BenchmarkComparisonResult& result);

    [[nodiscard]] BenchmarkState getState() const noexcept { return m_state; }
    [[nodiscard]] const std::string& getStatusText() const noexcept { return m_statusText; }
    [[nodiscard]] std::optional<BenchmarkComparisonResult>& getResults() noexcept { return m_results; }

    void requestCapture();
    void requestRecompute();
    void reset();

    void queueCommand(std::unique_ptr<IBenchmarkCommand> cmd);

    void startComputing(const cv::Mat& currentFrame, const common::SegmentationConfig& config);

    /**
     * @brief Polls the future to see if computation is done.
     */
    void poll();
};

} // namespace kmeans::io
