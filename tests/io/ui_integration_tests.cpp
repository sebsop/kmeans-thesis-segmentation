#include <atomic>
#include <mutex>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "io/benchmark_runner.hpp"
#include "io/ui_manager.hpp"

namespace ThesisTests::IO {

using namespace kmeans::io;
using namespace kmeans::common;

// A mock observer to test the UI notification system
class MockBenchmarkObserver : public IBenchmarkObserver {
  public:
    bool notified = false;
    void onBenchmarkComplete(const BenchmarkComparisonResult& result) override { notified = true; }
};

class IO_UIIntegration : public ::testing::Test {};

// 1. Benchmark Runner State Machine
TEST_F(IO_UIIntegration, BenchmarkStateTransitions) {
    BenchmarkRunner runner;
    EXPECT_EQ(runner.getState(), BenchmarkState::IDLE);

    runner.requestCapture();
    EXPECT_EQ(runner.getState(), BenchmarkState::CAPTURING);

    runner.reset();
    EXPECT_EQ(runner.getState(), BenchmarkState::IDLE);
}

// 2. UI Notification System (Observer Pattern)
TEST_F(IO_UIIntegration, ObserverNotification) {
    BenchmarkRunner runner;
    MockBenchmarkObserver observer;

    runner.addObserver(&observer);

    BenchmarkComparisonResult mockResult;
    runner.notifyObservers(mockResult);

    EXPECT_TRUE(observer.notified);

    runner.removeObserver(&observer);
    observer.notified = false;
    runner.notifyObservers(mockResult);
    EXPECT_FALSE(observer.notified);
}

// 3. UI Data Context Consistency
TEST_F(IO_UIIntegration, DataContextPropagatesConfig) {
    cv::Mat dummy;
    SegmentationConfig config;
    std::mutex mtx;
    bool showCentroids = false;
    std::atomic<bool> forceReset = false;
    BenchmarkRunner runner;

    UIDataContext ctx{.latestOriginal = dummy,
                      .latestSegmented = dummy,
                      .uiConfig = config,
                      .configMutex = mtx,
                      .showCentroids = showCentroids,
                      .forceReset = forceReset,
                      .currentWorkerFps = 60.0f,
                      .currentAlgoTimeMs = 16.0f,
                      .processedFrames = 100,
                      .benchmarkRunner = runner};

    // Simulate UI changing a value
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        ctx.uiConfig.k = 12;
    }

    // Verify the underlying config changed
    EXPECT_EQ(config.k, 12);
}

} // namespace ThesisTests::IO
