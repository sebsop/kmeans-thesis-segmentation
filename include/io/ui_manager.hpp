/**
 * @file ui_manager.hpp
 * @brief Main UI controller and OpenGL texture management.
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "io/benchmark_observer.hpp"
#include "io/benchmark_runner.hpp"

namespace kmeans::io {

/**
 * @struct TextureResource
 * @brief RAII wrapper for OpenGL texture lifetimes.
 *
 * Ensures that OpenGL textures are properly deleted from GPU memory when
 * the object goes out of scope.
 */
struct TextureResource {
    GLuint id = 0; ///< OpenGL Texture ID
    TextureResource() = default;
    ~TextureResource() noexcept {
        if (id != 0) {
            glDeleteTextures(1, &id);
        }
    }
    // Disable copy to prevent accidental texture deletion
    TextureResource(const TextureResource&) = delete;
    TextureResource& operator=(const TextureResource&) = delete;
};

/**
 * @struct UIDataContext
 * @brief Thread-safe context for passing data from the processing worker to the UI.
 *
 * This structure aggregates all dynamic data needed for rendering, including
 * shared OpenCV mats, configuration references, and performance metrics.
 */
struct UIDataContext {
    cv::Mat& latestOriginal;              ///< Reference to the most recent camera frame
    cv::Mat& latestSegmented;             ///< Reference to the most recent segmented frame
    common::SegmentationConfig& uiConfig; ///< Reference to the shared configuration
    std::mutex& configMutex;              ///< Mutex for thread-safe config access
    bool& showCentroids;                  ///< UI toggle for centroid overlay
    std::atomic<bool>& forceReset;        ///< Signal to worker to reset centers
    float currentWorkerFps;               ///< Processing speed (FPS)
    float currentAlgoTimeMs;              ///< Time taken per clustering iteration (ms)
    uint32_t processedFrames;             ///< Cumulative frame count
    BenchmarkRunner& benchmarkRunner;     ///< Handle to the background benchmarking system
};

namespace ui {
class ControlPanelUI;
class VideoFeedUI;
class BenchmarkOverlayUI;
} // namespace ui

/**
 * @class UIManager
 * @brief Coordinates the entire graphical user interface.
 *
 * UIManager is responsible for:
 * 1. Managing the conversion of OpenCV cv::Mat objects into OpenGL textures.
 * 2. Organizing the layout of the Control Panel, Video Feeds, and Benchmark Overlays.
 * 3. Implementing the custom premium visual theme (glassmorphism, modern colors).
 * 4. Observing benchmark progress and displaying results.
 */
class UIManager : public IBenchmarkObserver {
  private:
    TextureResource m_originalTexture;
    TextureResource m_segmentedTexture;
    TextureResource m_benchOriginalTexture;
    TextureResource m_benchClassicalTexture;
    TextureResource m_benchQuantumTexture;
    bool m_benchTexturesLoaded = false;

    // Sub-components (Strategy-like division of UI responsibilities)
    std::unique_ptr<ui::ControlPanelUI> m_controlPanel;
    std::unique_ptr<ui::VideoFeedUI> m_videoFeed;
    std::unique_ptr<ui::BenchmarkOverlayUI> m_benchmarkOverlay;

  public:
    UIManager();
    ~UIManager() override;

    /** @brief Configures the ImGui style with the project's custom premium theme. */
    static void applyPremiumTheme();

    /**
     * @brief The main render entry point called every frame.
     * @param ctx The data context containing fresh frames and metrics.
     */
    void render(UIDataContext& ctx);

    /** @brief Renders a static loading screen during application startup. */
    static void renderLoadingScreen(GLFWwindow* window);

    /** @brief Callback from the benchmark system to notify completion. */
    void onBenchmarkComplete(const BenchmarkComparisonResult& result) override;

  private:
    /** @brief Helper to upload an OpenCV Mat to a GPU-side OpenGL texture. */
    static void matToTexture(const cv::Mat& mat, TextureResource& textureRes);
};

} // namespace kmeans::io
