#include "io/benchmark_runner.hpp"

#include <opencv2/imgproc.hpp>

#include "clustering/clustering_manager.hpp"
#include "clustering/engines/kmeans_engine.hpp"
#include "common/constants.hpp"

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

void BenchmarkRunner::queueCommand(std::unique_ptr<IBenchmarkCommand> cmd) {
    m_commandQueue.push(std::move(cmd));
}

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

void BenchmarkRunner::addObserver(IBenchmarkObserver* observer) {
    m_observers.push_back(observer);
}

void BenchmarkRunner::removeObserver(IBenchmarkObserver* observer) {
    m_observers.erase(std::remove(m_observers.begin(), m_observers.end(), observer), m_observers.end());
}

void BenchmarkRunner::notifyObservers(const BenchmarkComparisonResult& result) {
    std::for_each(m_observers.begin(), m_observers.end(), [&](auto* obs) {
        if (obs) {
            obs->onBenchmarkComplete(result);
        }
    });
}

void BenchmarkRunner::poll() {
    if (m_state == BenchmarkState::COMPUTING) {
        if (!m_currentCommand && !m_commandQueue.empty()) {
            m_currentCommand = std::move(m_commandQueue.front());
            m_commandQueue.pop();
            m_currentCommand->execute();
        }

        if (m_currentCommand) {
            auto& fut = m_currentCommand->getFuture();
            if (fut.valid() && fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                m_results = fut.get();
                m_state = BenchmarkState::DONE;
                m_statusText = "Benchmark Complete.";
                notifyObservers(m_results.value());
                m_currentCommand.reset();

                [[maybe_unused]] size_t pendingCount = m_commandQueue.size();
                if (!m_commandQueue.empty()) {
                    m_state = BenchmarkState::COMPUTING;
                }
            }
        }
    }
}

} // namespace kmeans::io
