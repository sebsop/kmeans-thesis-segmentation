/**
 * @file control_panel_ui.hpp
 * @brief Left-side configuration and performance dashboard.
 */

#pragma once

#include "io/ui_manager.hpp"

namespace kmeans::io::ui {

/**
 * @class ControlPanelUI
 * @brief Renders the primary user interface for controlling the K-Means pipeline.
 *
 * This component provides:
 * 1. Hyper-parameter Sliders: Number of clusters (K), sampling stride, etc.
 * 2. Algorithm Selectors: Toggle between Classical and Quantum backends.
 * 3. Real-time Metrics: Live graphs for FPS and algorithm execution time.
 * 4. Benchmarking Controls: Trigger for the comparative analysis suite.
 */
class ControlPanelUI {
  public:
    ControlPanelUI() = default;
    ~ControlPanelUI() = default;

    /**
     * @brief Renders the control panel using ImGui.
     *
     * @param ctx The shared data context containing application state.
     * @param panelWidth The desired width of the panel in pixels.
     * @param benchTexturesLoaded State flag for benchmark texture availability.
     */
    static void render(UIDataContext& ctx, float panelWidth, bool& benchTexturesLoaded);
};

} // namespace kmeans::io::ui
