/**
 * @file benchmark_runner.cpp
 * @brief Implementation of the asynchronous benchmarking state machine.
 */

#include "io/benchmark_runner.hpp"

#include <opencv2/imgproc.hpp>

#include "clustering/clustering_manager.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::io {

/**
 * @brief Signals the system to capture the next available camera frame for analysis.
 */
void BenchmarkRunner::requestCapture() {
    m_state = BenchmarkState::CAPTURING;
    m_statusText = "Requesting frame from camera thread...";
}

/**
 * @brief Triggers a re-run of the comparison using the already captured frame.
 *
 * Useful when the user changes 'k' or 'stride' in the benchmark view and
 * wants to see the effect on the same data point.
 */
void BenchmarkRunner::requestRecompute() {
    m_state = BenchmarkState::RECOMPUTING;
}

/**
 * @brief Resets the runner to its idle state and clears result history.
 */
void BenchmarkRunner::reset() {
    m_state = BenchmarkState::IDLE;
    m_results.reset();
}

/**
 * @brief Queues a new benchmark task.
 * @param cmd Unique pointer to a concrete IBenchmarkCommand.
 */
void BenchmarkRunner::queueCommand(std::unique_ptr<IBenchmarkCommand> cmd) {
    m_commandQueue.push(std::move(cmd));
}

/**
 * @brief Transitions the runner into the computation phase.
 *
 * If recomputing, it reuses the existing original frame. Otherwise, it
 * clones the incoming live frame.
 *
 * @param currentFrame The latest frame from the live feed.
 * @param config The clustering parameters to use for the comparison.
 */
void BenchmarkRunner::startComputing(const cv::Mat& currentFrame, const common::SegmentationConfig& config) {
    bool isRecomputing = (m_state == BenchmarkState::RECOMPUTING);
    m_statusText = "Extracting frame and running dual-engine comparison...";

    cv::Mat benchFrame;
    if (isRecomputing && m_results.has_value()) {
        benchFrame = m_results->originalFrame.clone();
    } else {
        benchFrame = currentFrame.clone();
    }

    auto cmd = std::make_unique<RunBenchmarkCommand>(benchFrame, config);
    queueCommand(std::move(cmd));
    m_state = BenchmarkState::COMPUTING;
}

/**
 * @brief Registers a new result observer.
 */
void BenchmarkRunner::addObserver(IBenchmarkObserver* observer) {
    m_observers.push_back(observer);
}

/**
 * @brief Unregisters an existing result observer.
 */
void BenchmarkRunner::removeObserver(IBenchmarkObserver* observer) {
    auto [newEnd, _] = std::ranges::remove(m_observers, observer);
    m_observers.erase(newEnd, m_observers.end());
}

/**
 * @brief Broadcasts results to all registered observers.
 */
void BenchmarkRunner::notifyObservers(const BenchmarkComparisonResult& result) {
    std::ranges::for_each(m_observers, [&](auto* obs) {
        if (obs) {
            obs->onBenchmarkComplete(result);
        }
    });
}

/**
 * @brief Non-blocking check for command completion.
 *
 * This should be called once per UI frame. It manages the lifecycle of
 * asynchronous futures and transitions the state machine to DONE when
 * the worker thread finishes.
 */
void BenchmarkRunner::poll() {
    if (m_state == BenchmarkState::COMPUTING) {
        // Start the next command if the queue is not empty and no command is running
        if (!m_currentCommand && !m_commandQueue.empty()) {
            m_currentCommand = std::move(m_commandQueue.front());
            m_commandQueue.pop();
            m_currentCommand->execute();
        }

        // Check if the current asynchronous task is finished
        if (m_currentCommand) {
            auto& fut = m_currentCommand->getFuture();
            if (fut.valid() && fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                m_results = fut.get();
                m_state = BenchmarkState::DONE;
                m_statusText = "Benchmark Complete.";
                notifyObservers(m_results.value());
                m_currentCommand.reset();

                // If more tasks were queued (rare), continue computing
                if (!m_commandQueue.empty()) {
                    m_state = BenchmarkState::COMPUTING;
                }
            }
        }
    }
}

} // namespace kmeans::io
