#pragma once

#include "io/ui_manager.hpp"

namespace kmeans::io::ui {

/**
 * @brief Renders the left-side control panel containing hyperparameters,
 *        execution settings, and the performance dashboard.
 */
class ControlPanelUI {
  public:
    ControlPanelUI() = default;
    ~ControlPanelUI() = default;

    /**
     * @brief Renders the control panel using ImGui.
     * @param ctx The shared data context containing application state.
     * @param panelWidth The desired width of the panel.
     * @param benchTexturesLoaded Reference to the flag indicating if benchmark textures are loaded.
     */
    static void render(UIDataContext& ctx, float panelWidth, bool& benchTexturesLoaded);
};

} // namespace kmeans::io::ui
