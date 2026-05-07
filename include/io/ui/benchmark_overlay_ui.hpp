/**
 * @file benchmark_overlay_ui.hpp
 * @brief Formal head-to-head comparative analysis interface.
 */

#pragma once

#include "io/ui_manager.hpp"

namespace kmeans::io::ui {

/**
 * @class BenchmarkOverlayUI
 * @brief Renders the "Thesis Dashboard" for algorithm comparison.
 *
 * This component is used to display the results of a formal benchmark. It
 * presents a 3-way visual comparison (Original, Classical, Quantum) along
 * with a tabular breakdown of the mathematical quality metrics (WCSS,
 * Davies-Bouldin, Silhouette) and performance timings.
 */
class BenchmarkOverlayUI {
  public:
    BenchmarkOverlayUI() = default;
    ~BenchmarkOverlayUI() = default;

    /**
     * @brief Renders the full-screen benchmark analysis workspace.
     *
     * @param ctx Shared application state.
     * @param benchOriginalTex Static original frame used for the test.
     * @param benchClassicalTex Result of the Classical engine.
     * @param benchQuantumTex Result of the Quantum engine.
     * @param benchTexturesLoaded State flag for result persistence.
     * @param matToTexFunc Callback for uploading results to the GPU.
     */
    static void render(UIDataContext& ctx, TextureResource& benchOriginalTex, TextureResource& benchClassicalTex,
                       TextureResource& benchQuantumTex, bool& benchTexturesLoaded,
                       void (*matToTexFunc)(const cv::Mat&, TextureResource&));
};

} // namespace kmeans::io::ui
