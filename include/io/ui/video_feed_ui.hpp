/**
 * @file video_feed_ui.hpp
 * @brief Main workspace for viewing real-time segmentation.
 */

#pragma once

#include "io/ui_manager.hpp"

namespace kmeans::io::ui {

/**
 * @class VideoFeedUI
 * @brief Manages the visual display of the video segmentation pipeline.
 *
 * This component renders the raw camera input and the GPU-processed
 * segmentation result side-by-side. It also handles the optional
 * rendering of centroid "landmarks" on top of the segmented feed.
 */
class VideoFeedUI {
  public:
    VideoFeedUI() = default;
    ~VideoFeedUI() = default;

    /**
     * @brief Renders the video feed workspace using ImGui.
     *
     * @param ctx The shared data context.
     * @param panelWidth Offset from the left of the window (Control Panel width).
     * @param originalTex OpenGL resource for the raw feed.
     * @param segmentedTex OpenGL resource for the segmented result.
     * @param matToTexFunc Callback for uploading OpenCV data to the GPU.
     */
    static void render(UIDataContext& ctx, float panelWidth, TextureResource& originalTex,
                       TextureResource& segmentedTex, void (*matToTexFunc)(const cv::Mat&, TextureResource&));
};

} // namespace kmeans::io::ui
