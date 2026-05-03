#include "io/ui/control_panel_ui.hpp"

#include <algorithm>
#include <deque>
#include <imgui.h>
#include <numeric>

#include "common/constants.hpp"

namespace kmeans::io::ui {

void ControlPanelUI::render(UIDataContext& ctx, float panelWidth, bool& benchTexturesLoaded) {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, ImGui::GetIO().DisplaySize.y), ImGuiCond_Always);
    ImGui::Begin("Clustering Controls", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::SetCursorPosX(constants::UI_WINDOW_PADDING);

    {
        ImGuiStyle& style = ImGui::GetStyle();
        const float textH = ImGui::GetTextLineHeightWithSpacing();
        const float frameH = ImGui::GetFrameHeightWithSpacing();
        const float sepH = style.SeparatorSize;
        const float plotH = constants::UI_PLOT_HEIGHT;

        float contentH = 0.0f;
        contentH += textH;
        contentH += frameH * 2.0f;
        contentH += sepH;
        contentH += textH;
        contentH += frameH;
        contentH += frameH;
        contentH += sepH;
        contentH += textH;
        contentH += frameH;
        contentH += frameH;
        contentH += sepH * 2.0f;
        contentH += textH;
        contentH += textH;
        contentH += plotH;
        contentH += sepH;
        contentH += textH;
        contentH += frameH;

        const int gaps = constants::UI_LAYOUT_GAPS;
        contentH += gaps * style.ItemSpacing.y;
        contentH += style.WindowPadding.y * 2.0f;

        float availH = ImGui::GetContentRegionAvail().y;
        if (availH <= 0.0f) {
            availH = ImGui::GetWindowHeight() - style.WindowPadding.y * 2.0f;
        }

        float curY = ImGui::GetCursorPosY();
        float offset = std::max((availH - contentH) * 0.5f, 0.0f);
        ImGui::SetCursorPosY(curY + offset);
    }

    bool configChanged = false;
    common::SegmentationConfig pendingConfig;
    {
        std::scoped_lock<std::mutex> lock(ctx.configMutex);
        pendingConfig = ctx.uiConfig;
    }

    ImGui::Text("Core Hyperparameters");
    configChanged |= ImGui::SliderInt("Clusters (k)", &pendingConfig.k, constants::K_MIN, constants::K_MAX);
    configChanged |= ImGui::SliderInt("Learning Interval", &pendingConfig.learningInterval,
                                      constants::LEARN_INTERVAL_MIN, constants::LEARN_INTERVAL_MAX);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "How many frames to cache clusters before re-running K-Means. Set to 1 to force calculation every frame.");
    }

    ImGui::Separator();

    ImGui::Text("Architecture Strategy");

    configChanged |= ImGui::SliderInt("Stride", &pendingConfig.stride, 1, 16);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Downsample input data. Stride 1 = 100%% data, Stride 2 = 25%% data, Stride 4 = 6.25%% data.");
    }

    const char* engines[] = {"Classical (CPU)", "Quantum"};
    int currentEngine = (pendingConfig.algorithm == common::AlgorithmType::KMEANS_REGULAR) ? 0 : 1;
    if (ImGui::Combo("Execution Engine", &currentEngine, engines, 2)) {
        pendingConfig.algorithm =
            (currentEngine == 0) ? common::AlgorithmType::KMEANS_REGULAR : common::AlgorithmType::KMEANS_QUANTUM;
        configChanged = true;
    }

    if (configChanged) {
        std::scoped_lock<std::mutex> lock(ctx.configMutex);
        ctx.uiConfig = pendingConfig;
    }

    ImGui::Separator();
    ImGui::Text("Visualization Overlays");
    ImGui::Checkbox("Show Spatial Centroids", &ctx.showCentroids);
    if (ImGui::Button("Reset Centroids (Flush Memory)")) {
        ctx.forceReset = true;
    }

    ImGui::Separator();
    ImGui::Separator();
    ImGui::Text("Performance Dashboard");

    static float displayUIRenderFPS = 0.0f;
    static auto lastUIUpdateTime = std::chrono::high_resolution_clock::now();
    auto uiNow = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(uiNow - lastUIUpdateTime).count() >
        constants::UI_REFRESH_SLOW) {
        displayUIRenderFPS = ImGui::GetIO().Framerate;
        lastUIUpdateTime = uiNow;
    }
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "UI Render Speed: %.1f FPS", displayUIRenderFPS);

    static uint32_t lastProcessedFrames = 0;
    static std::deque<float> algoFpsHistory;

    float workerFps = ctx.currentWorkerFps;
    float algoTimeMs = ctx.currentAlgoTimeMs;
    uint32_t currentFrames = ctx.processedFrames;

    if (currentFrames != lastProcessedFrames) {
        if (algoFpsHistory.size() >= constants::FPS_HISTORY_WINDOW) {
            algoFpsHistory.pop_front();
        }
        algoFpsHistory.push_back(workerFps);
        lastProcessedFrames = currentFrames;
    }

    if (!algoFpsHistory.empty()) {
        auto [min_it, max_it] = std::minmax_element(algoFpsHistory.begin(), algoFpsHistory.end());
        float minFps = *min_it;
        float maxFps = *max_it;
        float sumFps = std::accumulate(algoFpsHistory.begin(), algoFpsHistory.end(), 0.0f);
        float avgFps = sumFps / static_cast<float>(algoFpsHistory.size());

        static float displayFps = 0.0f;
        static float displayMs = 0.0f;
        static float displayAvg = 0.0f;
        static auto lastUpdateTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTime).count() >
            constants::UI_REFRESH_FAST) {
            displayFps = workerFps;
            displayMs = algoTimeMs;
            displayAvg = avgFps;
            lastUpdateTime = now;
        }

        ImGui::Text("Algorithm Execution: %.2f ms", displayMs);
        ImGui::Text("Camera Pipeline: %.1f FPS", displayFps);
        ImGui::TextColored(ImVec4(constants::theme::SUCCESS_COL.r, constants::theme::SUCCESS_COL.g,
                                  constants::theme::SUCCESS_COL.b, constants::theme::SUCCESS_COL.a),
                           "Avg FPS: %.1f", displayAvg);

        int window = constants::UI_FPS_PLOT_WINDOW;
        std::vector<int> plot_indices(algoFpsHistory.size());
        std::iota(plot_indices.begin(), plot_indices.end(), 0);
        std::vector<float> fpsPlotBuf(algoFpsHistory.size());

        std::transform(plot_indices.begin(), plot_indices.end(), fpsPlotBuf.begin(), [&](int i) {
            int start = std::max(0, i - (window / 2));
            int end = std::min(static_cast<int>(algoFpsHistory.size()) - 1, i + (window / 2));
            float sum = std::accumulate(algoFpsHistory.begin() + start, algoFpsHistory.begin() + end + 1, 0.0f);
            return sum / static_cast<float>(end - start + 1);
        });

        ImGui::PlotLines("##Pipeline History", fpsPlotBuf.data(), static_cast<int>(fpsPlotBuf.size()), 0, nullptr, 0.0f,
                         (maxFps * 1.5f) + 5.0f, ImVec2(0, 60));
    }

    ImGui::Separator();
    ImGui::Text("Static Benchmarking");

    auto bState = ctx.benchmarkRunner.getState();
    if (bState == BenchmarkState::IDLE) {
        if (ImGui::Button("Capture & Run Comparison", ImVec2(-1, 30))) {
            ctx.benchmarkRunner.requestCapture();
        }
    } else if (bState == BenchmarkState::CAPTURING || bState == BenchmarkState::RECOMPUTING) {
        ImGui::TextColored(ImVec4(constants::theme::WARNING_COL.r, constants::theme::WARNING_COL.g,
                                  constants::theme::WARNING_COL.b, constants::theme::WARNING_COL.a),
                           "%s", ctx.benchmarkRunner.getStatusText().c_str());
    } else if (bState == BenchmarkState::COMPUTING) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.9f, 1.0f), "%s", ctx.benchmarkRunner.getStatusText().c_str());
        ctx.benchmarkRunner.poll();
        if (ctx.benchmarkRunner.getState() == BenchmarkState::DONE) {
            benchTexturesLoaded = false;
        }
    } else if (bState == BenchmarkState::DONE) {
        ImGui::TextColored(ImVec4(constants::theme::SUCCESS_COL.r, constants::theme::SUCCESS_COL.g,
                                  constants::theme::SUCCESS_COL.b, constants::theme::SUCCESS_COL.a),
                           "%s", ctx.benchmarkRunner.getStatusText().c_str());
        ImGui::Text("View the full-screen results.");
    }

    ImGui::End();
}

} // namespace kmeans::io::ui
