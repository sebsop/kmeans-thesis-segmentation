/**
 * @file benchmark_runner.hpp
 * @brief Orchestration for the comparative benchmarking engine.
 */

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

/**
 * @enum BenchmarkState
 * @brief Lifecycle states for a formal algorithm comparison.
 */
enum class BenchmarkState {
    IDLE,        ///< Waiting for user to trigger a test
    CAPTURING,   ///< Waiting for the next fresh camera frame to freeze
    RECOMPUTING, ///< Forcing a fresh calculation on an already captured frame
    COMPUTING,   ///< Background threads are running Classical and Quantum engines
    DONE         ///< Results are ready for display in the overlay
};

/**
 * @class BenchmarkRunner
 * @brief Manages the execution and lifecycle of head-to-head algorithm tests.
 *
 * The BenchmarkRunner implements a state-machine driven workflow that allows
 * users to capture a static scene and compare the performance and quality
 * of different K-Means implementations.
 *
 * Key Patterns used:
 * 1. Command Pattern: Long-running tasks are encapsulated as commands in a queue.
 * 2. Observer Pattern: UI components register as observers to be notified when
 *    a benchmark completes.
 * 3. State Machine: Transitions between capture, compute, and results states.
 */
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

    /** @brief Registers a listener for benchmark completion events. */
    void addObserver(IBenchmarkObserver* observer);

    /** @brief Unregisters a listener. */
    void removeObserver(IBenchmarkObserver* observer);

    /** @brief Broadcasts the final result to all registered observers. */
    void notifyObservers(const BenchmarkComparisonResult& result);

    /** @brief Returns the current internal state of the runner. */
    [[nodiscard]] BenchmarkState getState() const noexcept { return m_state; }

    /** @brief Returns a human-readable status message (e.g., "Computing Classical..."). */
    [[nodiscard]] const std::string& getStatusText() const noexcept { return m_statusText; }

    /** @brief Returns the comparison results if the state is DONE. */
    [[nodiscard]] std::optional<BenchmarkComparisonResult>& getResults() noexcept { return m_results; }

    /** @brief Transitions the runner to CAPTURING state to freeze the next frame. */
    void requestCapture();

    /** @brief Re-runs the benchmark using the currently captured frame and new config. */
    void requestRecompute();

    /** @brief Resets the runner to IDLE and clears all results. */
    void reset();

    /** @brief Adds a new operation to the internal command queue. */
    void queueCommand(std::unique_ptr<IBenchmarkCommand> cmd);

    /**
     * @brief Initiates the background computation for a benchmark.
     * @param currentFrame The frame to use for the test.
     * @param config The configuration to apply to both engines.
     */
    void startComputing(const cv::Mat& currentFrame, const common::SegmentationConfig& config);

    /**
     * @brief Non-blocking poll method called by the main loop to check for command completion.
     */
    void poll();
};

} // namespace kmeans::io
