#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <GLFW/glfw3.h>

#include "clustering/clustering_manager.hpp"
#include "io/benchmark_runner.hpp"
#include "io/ui_manager.hpp"

namespace kmeans::io {

class Application {
  private:
    GLFWwindow* m_window = nullptr;
    clustering::ClusteringManager m_manager;
    UIManager m_uiManager;
    BenchmarkRunner m_benchmarkRunner;

    // Multithreading Synchronization
    std::thread m_workerThread;
    std::atomic<bool> m_running{false};
    std::mutex m_dataMutex;
    std::mutex m_configMutex;

    cv::Mat m_latestOriginal;
    cv::Mat m_latestSegmented;
    std::vector<cv::Vec<float, 5>> m_latestCenters;
    common::SegmentationConfig m_uiConfig;

    std::chrono::high_resolution_clock::time_point m_lastWorkerTime;
    float m_currentWorkerFps = 0.0f;
    float m_currentAlgoTimeMs = 0.0f;
    uint32_t m_processedFrames = 0;

    // State initialized flags
    bool m_initialized = false;
    bool m_showCentroids = false;
    std::atomic<bool> m_forceReset{false};

    void initWindow();
    void initImGui();
    void cleanup() noexcept;

  public:
    Application();
    ~Application() noexcept;

    void run();
};

} // namespace kmeans::io
