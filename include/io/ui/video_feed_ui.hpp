#pragma once

#include "io/ui_manager.hpp"

namespace kmeans::io::ui {

/**
 * @brief Renders the main video segmentation feed showing original and clustered frames.
 */
class VideoFeedUI {
  public:
    VideoFeedUI() = default;
    ~VideoFeedUI() = default;

    /**
     * @brief Renders the video feed using ImGui.
     * @param ctx The shared data context containing application state.
     * @param panelWidth The width of the left-side panel.
     * @param originalTex The texture to draw the original frame to.
     * @param segmentedTex The texture to draw the segmented frame to.
     * @param matToTexFunc A function pointer to convert a cv::Mat to a TextureResource.
     */
    static void render(UIDataContext& ctx, float panelWidth, TextureResource& originalTex,
                       TextureResource& segmentedTex, void (*matToTexFunc)(const cv::Mat&, TextureResource&));
};

} // namespace kmeans::io::ui
