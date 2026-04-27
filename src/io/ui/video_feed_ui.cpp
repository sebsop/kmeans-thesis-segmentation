#include "io/ui/video_feed_ui.hpp"

#include <imgui.h>

namespace kmeans::io::ui {

void VideoFeedUI::render(UIDataContext& ctx, float panelWidth, TextureResource& originalTex,
                         TextureResource& segmentedTex, void (*matToTexFunc)(const cv::Mat&, TextureResource&)) {
    ImGui::SetNextWindowPos(ImVec2(panelWidth, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x - panelWidth, ImGui::GetIO().DisplaySize.y),
                             ImGuiCond_Always);
    ImGui::Begin("Video Segmentation Feed", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    if (!ctx.latestOriginal.empty()) {
        matToTexFunc(ctx.latestOriginal, originalTex);
        matToTexFunc(ctx.latestSegmented, segmentedTex);

        ImVec2 imgSize(static_cast<float>(ctx.latestOriginal.cols), static_cast<float>(ctx.latestOriginal.rows));
        ImVec2 segSize(static_cast<float>(ctx.latestSegmented.cols), static_cast<float>(ctx.latestSegmented.rows));

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
        ImU32 accentColor = ImGui::GetColorU32(ImVec4(0.60f, 0.40f, 0.90f, 1.00f));

        ImGui::GetWindowDrawList()->AddLine(ImVec2(startPos.x, startPos.y - 10.0f),
                                            ImVec2(startPos.x + totalWidth, startPos.y - 10.0f), accentColor, 5.0f);

        float contentHeight = ImGui::GetTextLineHeightWithSpacing() + imgSize.y;
        ImGui::GetWindowDrawList()->AddLine(ImVec2(startPos.x, startPos.y + contentHeight + 10.0f),
                                            ImVec2(startPos.x + totalWidth, startPos.y + contentHeight + 10.0f),
                                            accentColor, 5.0f);

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
