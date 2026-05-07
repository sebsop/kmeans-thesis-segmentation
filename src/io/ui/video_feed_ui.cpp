/**
 * @file video_feed_ui.cpp
 * @brief Implementation of the dual-pane video visualization UI.
 */

#include "io/ui/video_feed_ui.hpp"

#include <imgui.h>

#include "common/constants.hpp"

namespace kmeans::io::ui {

/**
 * @brief Renders the side-by-side "Original" vs "Segmented" video display.
 *
 * This method calculates dynamic offsets to keep the video feeds centered
 * in the remaining window space (excluding the side control panel). It
 * also adds decorative accent lines to the draw list to give it a more
 * "premium" feel.
 *
 * @param ctx Shared data context containing the latest frames.
 * @param panelWidth Width of the static side control panel.
 * @param originalTex OpenGL resource for the raw camera feed.
 * @param segmentedTex OpenGL resource for the K-Means output.
 * @param matToTexFunc Callback function to upload cv::Mat data to the GPU.
 */
void VideoFeedUI::render(UIDataContext& ctx, float panelWidth, TextureResource& originalTex,
                         TextureResource& segmentedTex, void (*matToTexFunc)(const cv::Mat&, TextureResource&)) {
    // 1. Position the video window to fill the rest of the screen
    ImGui::SetNextWindowPos(ImVec2(panelWidth, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x - panelWidth, ImGui::GetIO().DisplaySize.y),
                             ImGuiCond_Always);
    ImGui::Begin("Video Segmentation Feed", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    if (!ctx.latestOriginal.empty()) {
        // 2. Refresh GPU textures with the latest frames from the processing thread
        matToTexFunc(ctx.latestOriginal, originalTex);
        matToTexFunc(ctx.latestSegmented, segmentedTex);

        ImVec2 imgSize(static_cast<float>(ctx.latestOriginal.cols), static_cast<float>(ctx.latestOriginal.rows));
        ImVec2 segSize(static_cast<float>(ctx.latestSegmented.cols), static_cast<float>(ctx.latestSegmented.rows));

        // 3. Math for centering the feeds
        float totalWidth = imgSize.x + ImGui::GetStyle().ItemSpacing.x + segSize.x;
        float offsetX = (ImGui::GetWindowWidth() - totalWidth) * 0.5f;
        float offsetY = (ImGui::GetWindowHeight() - imgSize.y - ImGui::GetTextLineHeightWithSpacing()) * 0.5f;

        if (offsetX > 0) {
            ImGui::SetCursorPosX(offsetX);
        }
        if (offsetY > 0) {
            ImGui::SetCursorPosY(offsetY);
        }

        ImVec2 startPos = ImGui::GetCursorScreenPos();
        ImU32 accentColor = ImGui::GetColorU32(ImVec4(0.60f, 0.40f, 0.90f, 1.00f)); // Subtle purple accent

        // 4. Draw decorative borders using the ImGui DrawList
        ImGui::GetWindowDrawList()->AddLine(ImVec2(startPos.x, startPos.y - 10.0f),
                                            ImVec2(startPos.x + totalWidth, startPos.y - 10.0f), accentColor,
                                            constants::ui::BORDER_THICKNESS);

        float contentHeight = ImGui::GetTextLineHeightWithSpacing() + imgSize.y;
        ImGui::GetWindowDrawList()->AddLine(ImVec2(startPos.x, startPos.y + contentHeight + 10.0f),
                                            ImVec2(startPos.x + totalWidth, startPos.y + contentHeight + 10.0f),
                                            accentColor, constants::ui::BORDER_THICKNESS);

        // 5. Draw the actual images
        ImGui::BeginGroup();
        ImGui::Text("Original Frame");
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(originalTex.id)), imgSize, ImVec2(1, 0),
                     ImVec2(0, 1));
        ImGui::EndGroup();

        ImGui::SameLine();

        ImGui::BeginGroup();
        ImGui::Text("Clustered Frame");
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(segmentedTex.id)), segSize, ImVec2(1, 0),
                     ImVec2(0, 1));
        ImGui::EndGroup();
    } else {
        ImGui::Text("Warming up camera thread...");
    }

    ImGui::End();
}

} // namespace kmeans::io::ui
