#include "io/ui_manager.hpp"

#include <algorithm>
#include <cmath>
#include <deque>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <opencv2/imgproc.hpp>

#include "common/constants.hpp"
#include "io/ui/benchmark_overlay_ui.hpp"
#include "io/ui/control_panel_ui.hpp"
#include "io/ui/video_feed_ui.hpp"

// Windows <GL/gl.h> strictly supports OpenGL 1.1.
#ifndef GL_CLAMP_TO_EDGE
constexpr int GL_CLAMP_TO_EDGE = 0x812F;
#endif

namespace kmeans::io {

UIManager::UIManager()
    : m_controlPanel(std::make_unique<ui::ControlPanelUI>()),
      m_videoFeed(std::make_unique<ui::VideoFeedUI>()),
      m_benchmarkOverlay(std::make_unique<ui::BenchmarkOverlayUI>()) {}

UIManager::~UIManager() = default;

void UIManager::applyPremiumTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.12f, 0.94f);
    colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.60f, 0.40f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.60f, 0.40f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.70f, 0.50f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.60f, 0.40f, 0.90f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.70f, 0.50f, 1.00f, 1.00f);

    style.WindowRounding = 8.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.FramePadding = ImVec2(10.0f, 6.0f);
    style.WindowPadding = ImVec2(12.0f, 12.0f);
    style.ItemSpacing = ImVec2(8.0f, 8.0f);
}

void UIManager::matToTexture(const cv::Mat& mat, TextureResource& textureRes) {
    if (mat.empty()) {
        return;
    }

    cv::Mat rgbMat;
    cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGBA);

    if (textureRes.id == 0) {
        glGenTextures(1, &textureRes.id);
        glBindTexture(GL_TEXTURE_2D, textureRes.id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, textureRes.id);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgbMat.cols, rgbMat.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgbMat.ptr());
}

void UIManager::render(UIDataContext& ctx) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    float panelWidth = constants::UI_PANEL_WIDTH;

    m_controlPanel->render(ctx, panelWidth, m_benchTexturesLoaded);
    m_videoFeed->render(ctx, panelWidth, m_originalTexture, m_segmentedTexture, &UIManager::matToTexture);
    m_benchmarkOverlay->render(ctx, m_benchOriginalTexture, m_benchClassicalTexture, m_benchQuantumTexture, 
                               m_benchTexturesLoaded, &UIManager::matToTexture);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace kmeans::io
