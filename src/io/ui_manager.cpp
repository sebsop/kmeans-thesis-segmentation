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
    : m_controlPanel(std::make_unique<ui::ControlPanelUI>()), m_videoFeed(std::make_unique<ui::VideoFeedUI>()),
      m_benchmarkOverlay(std::make_unique<ui::BenchmarkOverlayUI>()) {}

UIManager::~UIManager() = default;

void UIManager::applyPremiumTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    auto toImVec4 = [](const constants::ColorRGBA& c) { return ImVec4(c.r, c.g, c.b, c.a); };

    colors[ImGuiCol_Text] = toImVec4(constants::theme::TEXT);
    colors[ImGuiCol_WindowBg] = toImVec4(constants::theme::WINDOW_BG);
    colors[ImGuiCol_PopupBg] = toImVec4(constants::theme::POPUP_BG);
    colors[ImGuiCol_Border] = toImVec4(constants::theme::BORDER);
    colors[ImGuiCol_FrameBg] = toImVec4(constants::theme::FRAME_BG);
    colors[ImGuiCol_FrameBgHovered] = toImVec4(constants::theme::FRAME_BG_HOVERED);
    colors[ImGuiCol_FrameBgActive] = toImVec4(constants::theme::FRAME_BG_ACTIVE);
    colors[ImGuiCol_TitleBg] = toImVec4(constants::theme::TITLE_BG);
    colors[ImGuiCol_TitleBgActive] = toImVec4(constants::theme::TITLE_BG_ACTIVE);
    colors[ImGuiCol_Button] = toImVec4(constants::theme::BUTTON);
    colors[ImGuiCol_ButtonHovered] = toImVec4(constants::theme::BUTTON_HOVERED);
    colors[ImGuiCol_ButtonActive] = toImVec4(constants::theme::BUTTON_ACTIVE);
    colors[ImGuiCol_Header] = toImVec4(constants::theme::HEADER);
    colors[ImGuiCol_HeaderHovered] = toImVec4(constants::theme::HEADER_HOVERED);
    colors[ImGuiCol_HeaderActive] = toImVec4(constants::theme::HEADER_ACTIVE);
    colors[ImGuiCol_CheckMark] = toImVec4(constants::theme::CHECK_MARK);
    colors[ImGuiCol_SliderGrab] = toImVec4(constants::theme::SLIDER_GRAB);
    colors[ImGuiCol_SliderGrabActive] = toImVec4(constants::theme::SLIDER_GRAB_ACTIVE);
    colors[ImGuiCol_PlotLines] = toImVec4(constants::theme::PLOT_LINES);
    colors[ImGuiCol_PlotLinesHovered] = toImVec4(constants::theme::PLOT_LINES_HOVERED);

    style.WindowRounding = constants::theme::WINDOW_ROUNDING;
    style.ChildRounding = constants::theme::CHILD_ROUNDING;
    style.FrameRounding = constants::theme::FRAME_ROUNDING;
    style.PopupRounding = constants::theme::POPUP_ROUNDING;
    style.GrabRounding = constants::theme::GRAB_ROUNDING;
    style.FramePadding = ImVec2(constants::theme::FRAME_PADDING_X, constants::theme::FRAME_PADDING_Y);
    style.WindowPadding = ImVec2(constants::theme::WINDOW_PADDING_X, constants::theme::WINDOW_PADDING_Y);
    style.ItemSpacing = ImVec2(constants::theme::ITEM_SPACING_X, constants::theme::ITEM_SPACING_Y);
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
