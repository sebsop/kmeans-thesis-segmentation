#include "io/ui/benchmark_overlay_ui.hpp"

#include <cmath>
#include <imgui.h>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "common/constants.hpp"

namespace kmeans::io::ui {

void BenchmarkOverlayUI::render(UIDataContext& ctx, TextureResource& benchOriginalTex,
                                TextureResource& benchClassicalTex, TextureResource& benchQuantumTex,
                                bool& benchTexturesLoaded, void (*matToTexFunc)(const cv::Mat&, TextureResource&)) {
    auto bState = ctx.benchmarkRunner.getState();
    auto& bResults = ctx.benchmarkRunner.getResults();

    if ((bState == BenchmarkState::DONE || bState == BenchmarkState::COMPUTING ||
         bState == BenchmarkState::RECOMPUTING) &&
        bResults) {
        if (!benchTexturesLoaded) {
            auto drawCentroids = [](cv::Mat& img, const std::vector<cv::Vec<float, 5>>& centers) {
                for (const auto& c : centers) {
                    cv::Point pt(static_cast<int>((c[3] / constants::SPATIAL_SCALE) * static_cast<float>(img.cols)),
                                 static_cast<int>((c[4] / constants::SPATIAL_SCALE) * static_cast<float>(img.rows)));
                    cv::Scalar color(c[0] / constants::COLOR_SCALE, c[1] / constants::COLOR_SCALE,
                                     c[2] / constants::COLOR_SCALE);
                    cv::circle(img, pt, constants::VIZ_CENTROID_RADIUS, color, -1);
                    cv::circle(img, pt, constants::VIZ_OUTLINE_WIDTH, cv::Scalar(255, 255, 255), 2);
                }
            };
            drawCentroids(bResults->classicalSegmented, bResults->classicalCenters);
            drawCentroids(bResults->quantumSegmented, bResults->quantumCenters);

            matToTexFunc(bResults->originalFrame, benchOriginalTex);
            matToTexFunc(bResults->classicalSegmented, benchClassicalTex);
            matToTexFunc(bResults->quantumSegmented, benchQuantumTex);
            benchTexturesLoaded = true;
        }

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Side-by-Side Algorithm Comparison", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoScrollbar);

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
            s.c1 = ImVec4(1, 1, 1, 1);
            s.c2 = ImVec4(1, 1, 1, 1);
            if (std::abs(v1 - v2) > 1e-5f) {
                bool v1Better = lowerIsBetter ? (v1 < v2) : (v1 > v2);
                s.c1 = v1Better ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                s.c2 = !v1Better ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                float worse = v1Better ? v2 : v1;
                float pct = std::abs(worse) > 1e-5f ? (std::abs(v1 - v2) / std::abs(worse)) * 100.0f : 0.0f;
                char pb[32];
                snprintf(pb, sizeof(pb), " (%.1f%% better)", pct);
                if (v1Better) {
                    s.t1 += pb;
                } else {
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

        if (ImGui::BeginTable("BenchTable", 3, ImGuiTableFlags_None)) {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float tableInnerW = avail.x;
            float spacing = ImGui::GetStyle().ItemSpacing.x;
            float colWidth = (tableInnerW - spacing * 2.0f) / 3.0f;
            const float imgScale = 0.825f;
            float ratio =
                static_cast<float>(bResults->originalFrame.rows) / static_cast<float>(bResults->originalFrame.cols);
            float imgW = colWidth * imgScale;
            float imgH = imgW * ratio;
            ImVec2 size(imgW, imgH);

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

            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Metrics Guide:");
            ImGui::Text("WCSS (Inertia): < Lower is better");
            ImGui::Text("Davies-Bouldin: < Lower is better");
            ImGui::Text("Silhouette: > Higher is better");
            ImGui::Text("Iterations: < Lower is better");
            ImGui::Text("Latency: < Lower is better");

            ImGui::TableSetColumnIndex(1);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Performance Metrics:");
            ImGui::TextColored(s_wcss.c1, "WCSS: %s", s_wcss.t1.c_str());
            ImGui::TextColored(s_db.c1, "Davies-Bouldin: %s", s_db.t1.c_str());
            ImGui::TextColored(s_sil.c1, "Approx Silhouette: %s", s_sil.t1.c_str());
            ImGui::TextColored(s_iter.c1, "Iterations: %s", s_iter.t1.c_str());
            ImGui::TextColored(s_lat.c1, "Latency: %s ms", s_lat.t1.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.8f, 1.0f), "Performance Metrics:");
            ImGui::TextColored(s_wcss.c2, "WCSS: %s", s_wcss.t2.c_str());
            ImGui::TextColored(s_db.c2, "Davies-Bouldin: %s", s_db.t2.c_str());
            ImGui::TextColored(s_sil.c2, "Approx Silhouette: %s", s_sil.t2.c_str());
            ImGui::TextColored(s_iter.c2, "Iterations: %s", s_iter.t2.c_str());
            ImGui::TextColored(s_lat.c2, "Latency: %s ms", s_lat.t2.c_str());

            ImGui::EndTable();
        }
        ImGui::Separator();

        float btnWidth = constants::UI_BTN_WIDTH_LG;
        float rerunWidth = constants::UI_BTN_WIDTH_MD;
        float buttonsTotalWidth = btnWidth + 20.0f + rerunWidth;

        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - buttonsTotalWidth) * 0.5f);

        if (ImGui::Button("Resume Live Feed", ImVec2(btnWidth, constants::UI_BTN_HEIGHT))) {
            ctx.benchmarkRunner.reset();
        }

        ImGui::SameLine(0, 20.0f);
        if (ImGui::Button("Rerun Frame", ImVec2(rerunWidth, constants::UI_BTN_HEIGHT))) {
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

        float kSliderWidth = 120.0f;
        float strideSliderWidth = 100.0f;
        float radioW = 260.0f;

        float kTextW = ImGui::CalcTextSize("K: ").x;
        float strideTextW = ImGui::CalcTextSize("Stride: ").x;
        float row2Width = kTextW + kSliderWidth + 30.0f + strideTextW + strideSliderWidth + 30.0f + radioW;

        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - row2Width) * 0.5f);
        ImGui::AlignTextToFramePadding();

        ImGui::Text("K:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(kSliderWidth);

        static bool s_needsRecompute = false;

        if (ImGui::SliderInt("##benchK", &tempK, constants::K_MIN, constants::K_MAX)) {
            std::scoped_lock<std::mutex> lock(ctx.configMutex);
            ctx.uiConfig.k = tempK;
            s_needsRecompute = true;
        }

        ImGui::SameLine(0, 30.0f);

        ImGui::Text("Stride:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(strideSliderWidth);
        if (ImGui::SliderInt("##benchStride", &tempStride, 1, 16)) {
            std::scoped_lock<std::mutex> lock(ctx.configMutex);
            ctx.uiConfig.stride = tempStride;
            s_needsRecompute = true;
        }

        ImGui::SameLine(0, 30.0f);

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

        if (s_needsRecompute && ctx.benchmarkRunner.getState() == BenchmarkState::DONE) {
            ctx.benchmarkRunner.requestRecompute();
            s_needsRecompute = false;
        }

        ImGui::End();
    }
}

} // namespace kmeans::io::ui
