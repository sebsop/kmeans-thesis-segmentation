#pragma once

#include "io/ui_manager.hpp"

namespace kmeans::io::ui {

/**
 * @brief Renders the full-screen side-by-side benchmark comparison overlay.
 */
class BenchmarkOverlayUI {
  public:
    BenchmarkOverlayUI() = default;
    ~BenchmarkOverlayUI() = default;

    /**
     * @brief Renders the benchmark overlay using ImGui.
     * @param ctx The shared data context containing application state.
     * @param benchOriginalTex Texture for original frame.
     * @param benchClassicalTex Texture for classical frame.
     * @param benchQuantumTex Texture for quantum frame.
     * @param benchTexturesLoaded Reference to the flag indicating if textures are loaded.
     * @param matToTexFunc Function pointer to convert cv::Mat to TextureResource.
     */
    void render(UIDataContext& ctx, TextureResource& benchOriginalTex, TextureResource& benchClassicalTex,
                TextureResource& benchQuantumTex, bool& benchTexturesLoaded,
                void (*matToTexFunc)(const cv::Mat&, TextureResource&));
};

} // namespace kmeans::io::ui
