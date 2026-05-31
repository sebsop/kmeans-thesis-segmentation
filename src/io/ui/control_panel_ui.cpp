/**
 * @file control_panel_ui.cpp
 * @brief Implementation of the primary sidebar control interface.
 */

#include "io/ui/control_panel_ui.hpp"

#include <algorithm>
#include <deque>
#include <imgui.h>
#include <numeric>

#include "common/constants.hpp"

namespace kmeans::io::ui {

static void showHelpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

/**
 * @brief Renders the side control panel with real-time parameter tuning and metrics.
 *
 * This method handles:
 * 1. Vertical centering of UI elements based on available region.
 * 2. Real-time configuration updates (k, stride, algorithm type).
 * 3. Performance tracking (FPS, latency, execution history plots).
 * 4. Benchmarking state orchestration.
 *
 * @param ctx The UI data context containing shared state.
 * @param panelWidth Fixed width for the sidebar.
 * @param benchTexturesLoaded [In/Out] Flag to trigger texture refresh when benchmarking completes.
 */
void ControlPanelUI::render(UIDataContext& ctx, float panelWidth, bool& benchTexturesLoaded) {
    // 1. Initial window setup (docked to the left)
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, ImGui::GetIO().DisplaySize.y), ImGuiCond_Always);
    ImGui::Begin("Clustering Controls", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::SetCursorPosX(constants::ui::WINDOW_PADDING);

    // 2. Vertical Centering Logic
    // We pre-calculate the height of all widgets to ensure the panel looks balanced.
    {
        const ImGuiStyle& style = ImGui::GetStyle();
        const float textH = ImGui::GetTextLineHeightWithSpacing();
        const float frameH = ImGui::GetFrameHeightWithSpacing();
        const float frameH_padded = ImGui::GetFontSize() + style.FramePadding.y * 1.5f * 2.0f + style.ItemSpacing.y;
        const float sepH = 2.0f + style.ItemSpacing.y;
        const float plotH = constants::ui::PLOT_HEIGHT + style.ItemSpacing.y;
        const float btnH = (constants::ui::BTN_HEIGHT * 0.5f) + style.ItemSpacing.y;

        float contentH = 0.0f;
        // 1. Core Hyperparameters
        contentH += textH;
        contentH += frameH_padded * 2.0f;
        // 2. Separator
        contentH += sepH;
        // 3. Architecture Strategy
        contentH += textH;
        contentH += frameH_padded * 2.0f;
        // 4. Separator
        contentH += sepH;
        // 5. Visualization Overlays
        contentH += textH;
        contentH += frameH;
        contentH += btnH;
        // 6. Separator
        contentH += sepH;
        // 7. Performance Dashboard
        contentH += textH;
        contentH += textH * 4.0f; // 4 lines of metrics text
        contentH += plotH;
        // 8. Separator
        contentH += sepH;
        // 9. Static Benchmarking
        contentH += textH;
        contentH += btnH;

        contentH += style.WindowPadding.y * 2.0f;

        float availH = ImGui::GetContentRegionAvail().y;
        if (availH <= 0.0f) {
            availH = ImGui::GetWindowHeight() - style.WindowPadding.y * 2.0f;
        }

        float curY = ImGui::GetCursorPosY();
        float offset = std::max((availH - contentH) * 0.5f, 0.0f);
        ImGui::SetCursorPosY(curY + offset);
    }

    // 3. Configuration Management (Hot-Swapping)
    bool configChanged = false;
    common::SegmentationConfig pendingConfig;
    {
        std::scoped_lock<std::mutex> lock(ctx.configMutex);
        pendingConfig = ctx.uiConfig;
    }

    ImGui::Text("Core Hyperparameters");
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, ImGui::GetStyle().FramePadding.y * 1.5f));
    configChanged |=
        ImGui::SliderInt("Clusters (k)", &pendingConfig.k, constants::clustering::K_MIN, constants::clustering::K_MAX);
    showHelpMarker("Number of centroids to calculate (number of distinct colors in the final segmented image).");
    configChanged |=
        ImGui::SliderInt("Learning Interval", &pendingConfig.learningInterval,
                         constants::clustering::LEARN_INTERVAL_MIN, constants::clustering::LEARN_INTERVAL_MAX);
    showHelpMarker("How many frames to cache clusters before re-running K-Means. Set to 1 to force calculation every frame.");
    ImGui::PopStyleVar();

    ImGui::Separator();

    ImGui::Text("Architecture Strategy");

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, ImGui::GetStyle().FramePadding.y * 1.5f));
    configChanged |= ImGui::SliderInt("Stride", &pendingConfig.stride, 1, 16);
    showHelpMarker("Downsample input data. Stride 1 = 100% data, Stride 2 = 25% data, Stride 4 = 6.25% data.");
    ImGui::PopStyleVar();

    const char* engines[] = {"Classical", "Quantum"};
    int currentEngine = (pendingConfig.algorithm == common::AlgorithmType::KMEANS_REGULAR) ? 0 : 1;
    
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, ImGui::GetStyle().FramePadding.y * 1.5f));
    if (ImGui::Combo("Execution Engine", &currentEngine, engines, 2)) {
        pendingConfig.algorithm =
            (currentEngine == 0) ? common::AlgorithmType::KMEANS_REGULAR : common::AlgorithmType::KMEANS_QUANTUM;
        configChanged = true;
    }
    ImGui::PopStyleVar();
    showHelpMarker("Select the processing backend: Classical CUDA K-Means or Quantum-inspired Hilbert space projected K-Means.");

    if (configChanged) {
        std::scoped_lock<std::mutex> lock(ctx.configMutex);
        ctx.uiConfig = pendingConfig;
        ctx.forceReset = true; // Force immediate recompute of K-Means
    }

    ImGui::Separator();
    ImGui::Text("Visualization Overlays");
    ImGui::Checkbox("Show Spatial Centroids", &ctx.showCentroids);
    showHelpMarker("Draw the computed 2D centroids directly onto the segmented frames.");
    if (ImGui::Button("Reset Centroids (Flush Memory)", ImVec2(-1, constants::ui::BTN_HEIGHT * 0.5f))) {
        ctx.forceReset = true;
    }

    ImGui::Separator();
    ImGui::Text("Performance Dashboard");

    // 5. Algorithm Pipeline Metrics
    static uint32_t lastProcessedFrames = 0;
    static std::deque<float> algoFpsHistory;

    float workerFps = ctx.currentWorkerFps;
    float algoTimeMs = ctx.currentAlgoTimeMs;
    float totalPipelineTimeMs = ctx.totalPipelineTimeMs;
    uint32_t currentFrames = ctx.processedFrames;

    float theoreticalFps = (algoTimeMs > 0.001f) ? (1000.0f / algoTimeMs) : 0.0f;
    float overallFps = (totalPipelineTimeMs > 0.001f) ? (1000.0f / totalPipelineTimeMs) : 0.0f;

    if (currentFrames != lastProcessedFrames) {
        if (algoFpsHistory.size() >= constants::ui::FPS_HISTORY_WINDOW) {
            algoFpsHistory.pop_front();
        }
        algoFpsHistory.push_back(overallFps);
        lastProcessedFrames = currentFrames;
    }

    if (!algoFpsHistory.empty()) {
        auto [min_it, max_it] = std::ranges::minmax_element(algoFpsHistory);
        float maxFps = *max_it;
        float sumFps = std::accumulate(algoFpsHistory.begin(), algoFpsHistory.end(), 0.0f);
        float avgFps = sumFps / static_cast<float>(algoFpsHistory.size());

        static float displayFps = 0.0f;
        static float displayMs = 0.0f;
        static float displayTotalMs = 0.0f;
        static float displayAvg = 0.0f;
        static float displayTheoretical = 0.0f;
        static auto lastUpdateTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTime).count() >
            constants::ui::REFRESH_FAST) {
            displayFps = workerFps;
            displayTotalMs = totalPipelineTimeMs;
            displayMs = algoTimeMs;
            displayTheoretical = theoreticalFps;
            displayAvg = avgFps;
            lastUpdateTime = now;
        }

        ImGui::Text("Core K-Means Latency: %.2f ms", displayMs);
        showHelpMarker("Time spent executing the selected clustering engine (Classical Lloyd's loop or Quantum-inspired phase-mapping) on the GPU (updates only on active learning frames).");

        ImGui::Text("Total Frame Latency: %.2f ms", displayTotalMs);
        showHelpMarker("Total time spent on GPU/CPU preprocessing, K-Means calculation (if active), and pixel assignment for the current frame.");

        ImGui::Text("Raw Engine Speed: %.1f FPS", displayTheoretical);
        showHelpMarker("The maximum throughput of the core clustering engine alone if it executed on every single frame without temporal caching.");

        ImGui::Text("Actual Frame Rate: %.1f FPS", displayFps);
        showHelpMarker("The active processing frame rate of the worker thread, capped by the camera's hardware capture rate.");

        ImGui::TextColored(ImVec4(constants::ui::theme::SUCCESS_COL.r, constants::ui::theme::SUCCESS_COL.g,
                                  constants::ui::theme::SUCCESS_COL.b, constants::ui::theme::SUCCESS_COL.a),
                           "Overall Throughput: %.1f FPS", displayAvg);
        showHelpMarker("Rolling average of the actual overall system processing throughput (1000ms / Total Frame Latency) across both cached and active frames.");

        // Render moving average history plot
        int window = constants::ui::FPS_PLOT_WINDOW;
        std::vector<int> plot_indices(algoFpsHistory.size());
        std::iota(plot_indices.begin(), plot_indices.end(), 0);
        std::vector<float> fpsPlotBuf(algoFpsHistory.size());

        std::ranges::transform(plot_indices, fpsPlotBuf.begin(), [&](int i) {
            int start = std::max(0, i - (window / 2));
            int end = std::min(static_cast<int>(algoFpsHistory.size()) - 1, i + (window / 2));
            float sum = std::accumulate(algoFpsHistory.begin() + start, algoFpsHistory.begin() + end + 1, 0.0f);
            return sum / static_cast<float>(end - start + 1);
        });

        float yMaxLimit = (maxFps * 1.2f) + 10.0f;
        float plotWidth = ImGui::GetContentRegionAvail().x - 140.0f; // Leave space for Y axis labels

        ImGui::BeginGroup();
        ImGui::PlotLines("##Theoretical Throughput History", fpsPlotBuf.data(), static_cast<int>(fpsPlotBuf.size()), 0, nullptr, 0.0f,
                         yMaxLimit, ImVec2(plotWidth, constants::ui::PLOT_HEIGHT));
        ImGui::EndGroup();

        ImGui::SameLine();

        ImGui::BeginGroup();
        ImGui::Text("%.0f FPS", yMaxLimit);
        ImGui::Dummy(ImVec2(0.0f, constants::ui::PLOT_HEIGHT - ImGui::GetTextLineHeightWithSpacing() * 2.0f));
        ImGui::Text("0 FPS");
        ImGui::EndGroup();
    }

    ImGui::Separator();
    ImGui::Text("Static Benchmarking");

    // 6. Benchmark Runner State Machine Integration
    auto bState = ctx.benchmarkRunner.getState();
    if (bState == BenchmarkState::IDLE) {
        if (ImGui::Button("Capture & Run Comparison", ImVec2(-1, constants::ui::BTN_HEIGHT * 0.5f))) {
            ctx.benchmarkRunner.requestCapture();
        }
    } else if (bState == BenchmarkState::CAPTURING || bState == BenchmarkState::RECOMPUTING) {
        ImGui::TextColored(ImVec4(constants::ui::theme::WARNING_COL.r, constants::ui::theme::WARNING_COL.g,
                                  constants::ui::theme::WARNING_COL.b, constants::ui::theme::WARNING_COL.a),
                           "%s", ctx.benchmarkRunner.getStatusText().c_str());
    } else if (bState == BenchmarkState::COMPUTING) {
        ImGui::TextColored(ImVec4(constants::ui::theme::ACCENT.r, constants::ui::theme::ACCENT.g,
                                  constants::ui::theme::ACCENT.b, constants::ui::theme::ACCENT.a),
                           "%s", ctx.benchmarkRunner.getStatusText().c_str());
        ctx.benchmarkRunner.poll();
        if (ctx.benchmarkRunner.getState() == BenchmarkState::DONE) {
            benchTexturesLoaded = false;
        }
    } else if (bState == BenchmarkState::DONE) {
        ImGui::TextColored(ImVec4(constants::ui::theme::SUCCESS_COL.r, constants::ui::theme::SUCCESS_COL.g,
                                  constants::ui::theme::SUCCESS_COL.b, constants::ui::theme::SUCCESS_COL.a),
                           "%s", ctx.benchmarkRunner.getStatusText().c_str());
        ImGui::Text("View the full-screen results.");
    }

    ImGui::End();
}

} // namespace kmeans::io::ui
