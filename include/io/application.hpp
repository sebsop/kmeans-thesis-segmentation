/**
 * @file application.hpp
 * @brief High-level orchestration for the K-Means segmentation application.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

#include "clustering/clustering_manager.hpp"
#include "io/benchmark_runner.hpp"
#include "io/ui_manager.hpp"

namespace kmeans::io {

/**
 * @class Application
 * @brief The central controller that manages the application lifecycle.
 *
 * Application is responsible for:
 * 1. Window Management: Initializing GLFW, OpenGL, and Dear ImGui.
 * 2. Thread Orchestration: Spawning and managing a dedicated "Worker Thread"
 *    that handles OpenCV camera capture and CUDA processing without blocking the UI.
 * 3. State Synchronization: Using thread-safe primitives (mutexes, atomics) to
 *    pass frames and metrics between the background worker and the foreground renderer.
 * 4. Exception Handling: Providing tiered exception safety guarantees for
 *    robust startup and shutdown.
 */
class Application {
  private:
    GLFWwindow* m_window = nullptr;          ///< The main OS window handle
    clustering::ClusteringManager m_manager; ///< The clustering pipeline Facade
    UIManager m_uiManager;                   ///< The UI component coordinator
    BenchmarkRunner m_benchmarkRunner;       ///< The background benchmark system

    // Multithreading Synchronization
    std::thread m_workerThread;         ///< Dedicated thread for heavy CUDA/OpenCV work
    std::atomic<bool> m_running{false}; ///< Global lifecycle flag
    std::mutex m_dataMutex;             ///< Protects shared frames and FPS metrics
    std::mutex m_configMutex;           ///< Protects the shared SegmentationConfig

    cv::Mat m_latestOriginal;                   ///< Latest frame from camera (shared)
    cv::Mat m_latestSegmented;                  ///< Latest processed frame (shared)
    std::vector<FeatureVector> m_latestCenters; ///< Latest calculated centroids (shared)
    common::SegmentationConfig m_uiConfig;      ///< Current UI-side configuration state

    std::chrono::high_resolution_clock::time_point m_lastWorkerTime;
    float m_currentWorkerFps = 0.0f;
    float m_currentAlgoTimeMs = 0.0f;
    uint32_t m_processedFrames = 0;

    // State flags
    bool m_initialized = false;            ///< True when the first frame is ready
    bool m_showCentroids = false;          ///< Toggle for landmark overlay
    std::atomic<bool> m_forceReset{false}; ///< Signal to re-seed centroids

    common::SegmentationConfig m_savedBenchConfig; ///< Stashed config during benchmark
    bool m_benchWasActive = false;                 ///< State tracker for benchmark mode transition

    /** @brief Initializes the GLFW window and OpenGL context. */
    void initWindow();
    /** @brief Initializes the Dear ImGui context and backends. */
    void initImGui();
    /** @brief Performs RAII-compliant cleanup of all resources. */
    void cleanup() noexcept;

  public:
    /** @brief Constructs the application and initializes subsystems. */
    Application();
    /** @brief Safely shuts down the worker thread and window. */
    ~Application() noexcept;

    /**
     * @brief Starts the main application loop.
     *
     * This method enters a blocking loop that handles window events and
     * UI rendering until the window is closed.
     */
    void run();
};

} // namespace kmeans::io
