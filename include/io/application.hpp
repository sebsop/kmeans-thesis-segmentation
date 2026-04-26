#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

// Include GLFW wrapper
#include <chrono>

#include <GLFW/glfw3.h>

#include "clustering/clustering_manager.hpp"
#include "clustering/metrics.hpp"
#include <future>
#include <optional>

namespace kmeans::io {

// RAII explicit wrapper around OpenGL texture pointers to guarantee VRAM deallocation
struct TextureResource {
    GLuint id = 0;
    TextureResource() = default;
    ~TextureResource() noexcept {
        if (id != 0) {
            glDeleteTextures(1, &id);
        }
    }
    // Disable copy
    TextureResource(const TextureResource&) = delete;
    TextureResource& operator=(const TextureResource&) = delete;
};

class Application {
  private:
    GLFWwindow* m_window = nullptr;
    clustering::ClusteringManager m_manager;

    // Automatically managed OpenGL Texture handles
    TextureResource m_originalTexture;
    TextureResource m_segmentedTexture;

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

    // Benchmark State
    struct BenchmarkComparisonResult {
        cv::Mat originalFrame;
        cv::Mat classicalSegmented;
        cv::Mat quantumSegmented;
        clustering::metrics::BenchmarkResults classicalMetrics;
        clustering::metrics::BenchmarkResults quantumMetrics;
        std::vector<cv::Vec<float, 5>> classicalCenters;
        std::vector<cv::Vec<float, 5>> quantumCenters;
    };

    enum class BenchmarkState { IDLE, CAPTURING, COMPUTING, DONE };
    BenchmarkState m_benchmarkState = BenchmarkState::IDLE;
    std::future<BenchmarkComparisonResult> m_benchmarkFuture;
    std::optional<BenchmarkComparisonResult> m_benchmarkResults;
    std::string m_benchmarkStatusText;

    TextureResource m_benchOriginalTexture;
    TextureResource m_benchClassicalTexture;
    TextureResource m_benchQuantumTexture;
    bool m_benchTexturesLoaded = false;
    
    // UI Theming
    void applyPremiumTheme();

    void initWindow();
    void initImGui();
    void cleanup() noexcept;

    void renderUI();
    static void matToTexture(const cv::Mat& mat, TextureResource& textureRes);

  public:
    Application();
    ~Application() noexcept;

    void run();
};

} // namespace kmeans::io
