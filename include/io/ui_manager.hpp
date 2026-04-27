#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "io/benchmark_runner.hpp"

namespace kmeans::io {

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

struct UIDataContext {
    cv::Mat& latestOriginal;
    cv::Mat& latestSegmented;
    common::SegmentationConfig& uiConfig;
    std::mutex& configMutex;
    bool& showCentroids;
    std::atomic<bool>& forceReset;
    float currentWorkerFps;
    float currentAlgoTimeMs;
    uint32_t processedFrames;
    BenchmarkRunner& benchmarkRunner;
};

// Forward declare the UI components (to avoid huge include trees)
namespace ui {
class ControlPanelUI;
class VideoFeedUI;
class BenchmarkOverlayUI;
} // namespace ui

class UIManager {
  private:
    TextureResource m_originalTexture;
    TextureResource m_segmentedTexture;
    TextureResource m_benchOriginalTexture;
    TextureResource m_benchClassicalTexture;
    TextureResource m_benchQuantumTexture;
    bool m_benchTexturesLoaded = false;

    // Sub-components
    std::unique_ptr<ui::ControlPanelUI> m_controlPanel;
    std::unique_ptr<ui::VideoFeedUI> m_videoFeed;
    std::unique_ptr<ui::BenchmarkOverlayUI> m_benchmarkOverlay;

  public:
    UIManager();
    ~UIManager();

    static void applyPremiumTheme();

    /**
     * @brief Renders the complete ImGui interface using the provided context.
     */
    void render(UIDataContext& ctx);

  private:
    static void matToTexture(const cv::Mat& mat, TextureResource& textureRes);
};

} // namespace kmeans::io
