/**
 * @file benchmark_overlay_ui.cpp
 * @brief Implementation of the full-screen algorithm comparison dashboard.
 */

#include "io/ui/benchmark_overlay_ui.hpp"

#include <cmath>
#include <imgui.h>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "common/constants.hpp"

namespace kmeans::io::ui {

/**
 * @brief Renders a full-screen comparison table between Classical and Quantum backends.
 *
 * This method is the "Thesis Summary" view. It displays:
 * 1. Side-by-side segmented images with spatial centroids drawn on top.
 * 2. A mathematical scorecard comparing WCSS, Davies-Bouldin, Silhouette, Iterations, and Latency.
 * 3. Color-coded results (Green for better, Red for worse) with percentage improvements.
 * 4. Interactive "Recompute" controls to tweak parameters (k, stride) on the exact same captured frame.
 *
 * @param ctx The UI data context.
 * @param benchOriginalTex GPU resource for the captured raw frame.
 * @param benchClassicalTex GPU resource for the classical result.
 * @param benchQuantumTex GPU resource for the quantum result.
 * @param benchTexturesLoaded [In/Out] Manages GPU upload state.
 * @param matToTexFunc Callback for texture uploading.
 */
void BenchmarkOverlayUI::render(UIDataContext& ctx, TextureResource& benchOriginalTex,
                                TextureResource& benchClassicalTex, TextureResource& benchQuantumTex,
                                bool& benchTexturesLoaded, void (*matToTexFunc)(const cv::Mat&, TextureResource&)) {
    auto bState = ctx.benchmarkRunner.getState();
    auto& bResults = ctx.benchmarkRunner.getResults();

    // 1. Only render if we have a valid benchmark state
    if ((bState == BenchmarkState::DONE || bState == BenchmarkState::COMPUTING ||
         bState == BenchmarkState::RECOMPUTING) &&
        bResults) {

        // 2. Texture Preparation
        // If results just arrived, we draw centroids onto the cv::Mat results and upload to GPU.
        if (!benchTexturesLoaded) {
            auto drawCentroids = [](cv::Mat& img, const std::vector<FeatureVector>& centers) {
                std::ranges::for_each(centers, [&](const auto& c) {
                    // Map spatial features (normalized) back to pixel coordinates
                    cv::Point pt(
                        static_cast<int>((c[3] / constants::video::SPATIAL_SCALE) * static_cast<float>(img.cols)),
                        static_cast<int>((c[4] / constants::video::SPATIAL_SCALE) * static_cast<float>(img.rows)));
                    cv::Scalar color(c[0] / constants::video::COLOR_SCALE, c[1] / constants::video::COLOR_SCALE,
                                     c[2] / constants::video::COLOR_SCALE);

                    // Draw a clean centroid with an outline for contrast
                    cv::circle(img, pt, constants::viz::CENTROID_RADIUS, color, -1);
                    cv::circle(img, pt, constants::viz::OUTLINE_WIDTH, cv::Scalar(255, 255, 255), 2);
                });
            };
            drawCentroids(bResults->classicalSegmented, bResults->classicalCenters);
            drawCentroids(bResults->quantumSegmented, bResults->quantumCenters);

            matToTexFunc(bResults->originalFrame, benchOriginalTex);
            matToTexFunc(bResults->classicalSegmented, benchClassicalTex);
            matToTexFunc(bResults->quantumSegmented, benchQuantumTex);
            benchTexturesLoaded = true;
        }

        // 3. Window Layout
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Side-by-Side Algorithm Comparison", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoScrollbar);

        // 4. Mathematical Comparison Helper
        // This lambda logic handles the "Who won?" styling for each metric.
        struct MetricStyle {
            ImVec4 c1;
            ImVec4 c2;
            std::string t1;
            std::string t2;
        };
        auto getStyle = [](float v1, float v2, bool lowerIsBetter, bool isInt = false) -> MetricStyle {
            MetricStyle s;
            char b1[64];
            char b2[64];
            if (isInt) {
                snprintf(b1, sizeof(b1), "%d", static_cast<int>(v1));
                snprintf(b2, sizeof(b2), "%d", static_cast<int>(v2));
            } else if (v1 > 10.0f || v2 > 10.0f) {
                snprintf(b1, sizeof(b1), "%.2f", v1);
                snprintf(b2, sizeof(b2), "%.2f", v2);
            } else {
                snprintf(b1, sizeof(b1), "%.4f", v1);
                snprintf(b2, sizeof(b2), "%.4f", v2);
            }
            s.t1 = b1;
            s.t2 = b2;
            s.c1 = ImVec4(constants::ui::theme::NEUTRAL_COL.r, constants::ui::theme::NEUTRAL_COL.g,
                          constants::ui::theme::NEUTRAL_COL.b, constants::ui::theme::NEUTRAL_COL.a);
            s.c2 = s.c1;

            if (std::abs(v1 - v2) > constants::math::EPSILON) {
                bool v1Better = lowerIsBetter ? (v1 < v2) : (v1 > v2);
                ImVec4 success = ImVec4(constants::ui::theme::SUCCESS_COL.r, constants::ui::theme::SUCCESS_COL.g,
                                        constants::ui::theme::SUCCESS_COL.b, constants::ui::theme::SUCCESS_COL.a);
                ImVec4 error = ImVec4(constants::ui::theme::ERROR_COL.r, constants::ui::theme::ERROR_COL.g,
                                      constants::ui::theme::ERROR_COL.b, constants::ui::theme::ERROR_COL.a);

                s.c1 = v1Better ? success : error;
                s.c2 = !v1Better ? success : error;

                float worse = v1Better ? v2 : v1;
                float pct =
                    std::abs(worse) > constants::math::EPSILON ? (std::abs(v1 - v2) / std::abs(worse)) * 100.0f : 0.0f;
                char pb[32];
                snprintf(pb, sizeof(pb), " (%.1f%% better)", pct);
                if (v1Better) {
                    s.t1 += pb;
                }
                else {
                    s.t2 += pb;
                }
            }
            return s;
        };

        const auto& cm = bResults->classicalMetrics;
        const auto& qm = bResults->quantumMetrics;
        auto s_wcss = getStyle(cm.wcss, qm.wcss, true);
        auto s_db = getStyle(cm.daviesBouldin, qm.daviesBouldin, true);
        auto s_sil = getStyle(cm.silhouetteScore, qm.silhouetteScore, false);
        auto s_iter = getStyle(static_cast<float>(cm.iterations), static_cast<float>(qm.iterations), true, true);
        auto s_lat = getStyle(cm.executionTimeMs, qm.executionTimeMs, true);

        // 5. Comparison Table Rendering
        if (ImGui::BeginTable("BenchTable", constants::ui::BENCH_COL_COUNT, ImGuiTableFlags_None)) {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float tableInnerW = avail.x;
            float spacing = ImGui::GetStyle().ItemSpacing.x;
            float colWidth = (tableInnerW - spacing * 2.0f) / static_cast<float>(constants::ui::BENCH_COL_COUNT);
            const float imgScale = constants::ui::BENCH_IMG_SCALE;
            float ratio =
                static_cast<float>(bResults->originalFrame.rows) / static_cast<float>(bResults->originalFrame.cols);
            float imgW = colWidth * imgScale;
            float imgH = imgW * ratio;
            ImVec2 size(imgW, imgH);

            // Row 1: Visual Results
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            {
                const char* title = "1. Original Frame";
                float curX = ImGui::GetCursorPosX();
                float textW = ImGui::CalcTextSize(title).x;
                ImGui::SetCursorPosX(curX + ((colWidth - textW) * 0.5f));
                ImGui::Text("%s", title);
                ImGui::SetCursorPosX(curX + ((colWidth - imgW) * 0.5f));
                ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(benchOriginalTex.id)), size, ImVec2(1, 0),
                             ImVec2(0, 1));
            }

            ImGui::TableSetColumnIndex(1);
            {
                const char* title = "2. Classical K-Means";
                float curX = ImGui::GetCursorPosX();
                float textW = ImGui::CalcTextSize(title).x;
                ImGui::SetCursorPosX(curX + ((colWidth - textW) * 0.5f));
                ImGui::Text("%s", title);
                ImGui::SetCursorPosX(curX + ((colWidth - imgW) * 0.5f));
                ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(benchClassicalTex.id)), size, ImVec2(1, 0),
                             ImVec2(0, 1));
            }

            ImGui::TableSetColumnIndex(2);
            {
                const char* title = "3. Quantum K-Means";
                float curX = ImGui::GetCursorPosX();
                float textW = ImGui::CalcTextSize(title).x;
                ImGui::SetCursorPosX(curX + ((colWidth - textW) * 0.5f));
                ImGui::Text("%s", title);
                ImGui::SetCursorPosX(curX + ((colWidth - imgW) * 0.5f));
                ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(benchQuantumTex.id)), size, ImVec2(1, 0),
                             ImVec2(0, 1));
            }

            // Row 2: Mathematical Metrics
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(constants::ui::theme::BENCH_GUIDE.r, constants::ui::theme::BENCH_GUIDE.g,
                                      constants::ui::theme::BENCH_GUIDE.b, constants::ui::theme::BENCH_GUIDE.a),
                               "Metrics Guide:");
            ImGui::Text("WCSS (Inertia): < Lower is better");
            ImGui::Text("Davies-Bouldin: < Lower is better");
            ImGui::Text("Silhouette: > Higher is better");
            ImGui::Text("Iterations: < Lower is better");
            ImGui::Text("Latency: < Lower is better");

            ImGui::TableSetColumnIndex(1);
            ImGui::Separator();
            ImGui::TextColored(
                ImVec4(constants::ui::theme::BENCH_TITLE_CLASSICAL.r, constants::ui::theme::BENCH_TITLE_CLASSICAL.g,
                       constants::ui::theme::BENCH_TITLE_CLASSICAL.b, constants::ui::theme::BENCH_TITLE_CLASSICAL.a),
                "Performance Metrics:");
            ImGui::TextColored(s_wcss.c1, "WCSS: %s", s_wcss.t1.c_str());
            ImGui::TextColored(s_db.c1, "Davies-Bouldin: %s", s_db.t1.c_str());
            ImGui::TextColored(s_sil.c1, "Approx Silhouette: %s", s_sil.t1.c_str());
            ImGui::TextColored(s_iter.c1, "Iterations: %s", s_iter.t1.c_str());
            ImGui::TextColored(s_lat.c1, "Latency: %s ms", s_lat.t1.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::Separator();
            ImGui::TextColored(
                ImVec4(constants::ui::theme::BENCH_TITLE_QUANTUM.r, constants::ui::theme::BENCH_TITLE_QUANTUM.g,
                       constants::ui::theme::BENCH_TITLE_QUANTUM.b, constants::ui::theme::BENCH_TITLE_QUANTUM.a),
                "Performance Metrics:");
            ImGui::TextColored(s_wcss.c2, "WCSS: %s", s_wcss.t2.c_str());
            ImGui::TextColored(s_db.c2, "Davies-Bouldin: %s", s_db.t2.c_str());
            ImGui::TextColored(s_sil.c2, "Approx Silhouette: %s", s_sil.t2.c_str());
            ImGui::TextColored(s_iter.c2, "Iterations: %s", s_iter.t2.c_str());
            ImGui::TextColored(s_lat.c2, "Latency: %s ms", s_lat.t2.c_str());

            ImGui::EndTable();
        }
        ImGui::Separator();

        // 6. Navigation and Recompute Controls
        float btnWidth = constants::ui::BTN_WIDTH_LG;
        float rerunWidth = constants::ui::BTN_WIDTH_MD;
        float buttonsTotalWidth = btnWidth + constants::ui::BENCH_BTN_PADDING + rerunWidth;

        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - buttonsTotalWidth) * 0.5f);

        if (ImGui::Button("Resume Live Feed", ImVec2(btnWidth, constants::ui::BTN_HEIGHT))) {
            ctx.benchmarkRunner.reset();
        }

        ImGui::SameLine(0, constants::ui::BENCH_BTN_PADDING);
        if (ImGui::Button("Rerun Frame", ImVec2(rerunWidth, constants::ui::BTN_HEIGHT))) {
            ctx.benchmarkRunner.requestRecompute();
        }

        ImGui::Dummy(ImVec2(0.0f, 10.0f));

        int tempK = 0;
        int tempStride = 0;
        common::InitializationType currentInitType;
        {
            std::scoped_lock<std::mutex> lock(ctx.configMutex);
            tempK = ctx.uiConfig.k;
            tempStride = ctx.uiConfig.stride;
            currentInitType = ctx.uiConfig.init;
        }

        float kSliderWidth = constants::ui::K_SLIDER_WIDTH;
        float strideSliderWidth = constants::ui::STRIDE_SLIDER_WIDTH;
        float radioW = constants::ui::RADIO_WIDTH;

        float kTextW = ImGui::CalcTextSize("K: ").x;
        float strideTextW = ImGui::CalcTextSize("Stride: ").x;
        float row2Width = kTextW + kSliderWidth + constants::ui::BENCH_SLIDER_SPACING + strideTextW +
                          strideSliderWidth + constants::ui::BENCH_SLIDER_SPACING + radioW;

        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - row2Width) * 0.5f);
        ImGui::AlignTextToFramePadding();

        ImGui::Text("K:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(kSliderWidth);

        static bool s_needsRecompute = false;

        if (ImGui::SliderInt("##benchK", &tempK, constants::clustering::K_MIN, constants::clustering::K_MAX)) {
            std::scoped_lock<std::mutex> lock(ctx.configMutex);
            ctx.uiConfig.k = tempK;
            s_needsRecompute = true;
        }

        ImGui::SameLine(0, constants::ui::BENCH_SLIDER_SPACING);

        ImGui::Text("Stride:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(strideSliderWidth);
        if (ImGui::SliderInt("##benchStride", &tempStride, 1, 16)) {
            std::scoped_lock<std::mutex> lock(ctx.configMutex);
            ctx.uiConfig.stride = tempStride;
            s_needsRecompute = true;
        }

        ImGui::SameLine(0, constants::ui::BENCH_SLIDER_SPACING);

        int currentInit = (currentInitType == common::InitializationType::KMEANS_PLUSPLUS) ? 0 : 1;
        int oldInit = currentInit;
        ImGui::Text("Init Strategy:");
        ImGui::SameLine();
        ImGui::RadioButton("K-Means++", &currentInit, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Random", &currentInit, 1);

        if (currentInit != oldInit) {
            std::scoped_lock<std::mutex> lock(ctx.configMutex);
            ctx.uiConfig.init =
                (currentInit == 0) ? common::InitializationType::KMEANS_PLUSPLUS : common::InitializationType::RANDOM;
            s_needsRecompute = true;
        }

        // Automatic re-trigger if parameters changed while in benchmark view
        if (s_needsRecompute && ctx.benchmarkRunner.getState() == BenchmarkState::DONE) {
            ctx.benchmarkRunner.requestRecompute();
            s_needsRecompute = false;
        }

        ImGui::End();
    }
}

} // namespace kmeans::io::ui
