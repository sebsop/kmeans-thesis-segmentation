#include "io/application.hpp"

#include <deque>

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Backend configurations
#include "common/config.hpp"
#include "common/constants.hpp"
#include "common/enums.hpp"

// Windows <GL/gl.h> strictly supports OpenGL 1.1.
#ifndef GL_CLAMP_TO_EDGE
constexpr int GL_CLAMP_TO_EDGE = 0x812F;
#endif

namespace kmeans::io {

Application::Application() : m_uiConfig(m_manager.getConfig()), m_initialized(true) {
    initWindow();
    initImGui();
}

Application::~Application() noexcept {
    if (m_initialized) {
        cleanup();
    }
}

void Application::initWindow() {
    if (glfwInit() == 0) {
        std::cerr << "Failed to initialize GLFW. Application cannot start.\n";
        exit(EXIT_FAILURE);
    }

    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(1400, 438, "K-Means Segmentation Thesis - ImGui Dashboard", nullptr, nullptr);
    if (m_window == nullptr) {
        std::cerr << "Failed to create window using GLFW.\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0); // Disable vsync to uncap from monitor refresh rate
}

void Application::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    applyPremiumTheme();

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

void Application::cleanup() noexcept {
    // Textures are automatically deleted by the TextureResource RAII destructor
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Application::applyPremiumTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Premium Dark / Glassmorphism theme
    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.12f, 0.12f, 0.12f, 0.94f);
    colors[ImGuiCol_Border]                 = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.24f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.60f, 0.40f, 0.90f, 1.00f); // Vibrant purple accent
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.60f, 0.40f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.70f, 0.50f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.60f, 0.40f, 0.90f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.70f, 0.50f, 1.00f, 1.00f);

    style.WindowRounding    = 8.0f;
    style.ChildRounding     = 6.0f;
    style.FrameRounding     = 6.0f;
    style.PopupRounding     = 6.0f;
    style.GrabRounding      = 6.0f;
    style.FramePadding      = ImVec2(10.0f, 6.0f);
    style.WindowPadding     = ImVec2(12.0f, 12.0f);
    style.ItemSpacing       = ImVec2(8.0f, 8.0f);
}

void Application::matToTexture(const cv::Mat& mat, TextureResource& textureRes) {
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

void Application::renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 1. Control Panel
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(constants::UI_PANEL_WIDTH, ImGui::GetIO().DisplaySize.y), ImGuiCond_Always);
    ImGui::Begin("Clustering Controls", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    bool configChanged = false;

    // Work on a local copy so ImGui writes never race with the worker thread
    common::SegmentationConfig pendingConfig = m_uiConfig;

    ImGui::Text("Core Hyperparameters");
    configChanged |= ImGui::SliderInt("Clusters (k)", &pendingConfig.k, constants::K_MIN, constants::K_MAX);
    configChanged |= ImGui::SliderInt("Learning Interval", &pendingConfig.learningInterval, constants::LEARN_INTERVAL_MIN,
                                      constants::LEARN_INTERVAL_MAX);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "How many frames to cache clusters before re-running K-Means. Set to 1 to force calculation every frame.");
    }

    ImGui::Separator();

    ImGui::Text("Architecture Strategy");

    configChanged |= ImGui::SliderInt("Stride", &pendingConfig.stride, 1, 16);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Downsample input data. Stride 1 = 100%% data, Stride 2 = 25%% data, Stride 4 = 6.25%% data.");
    }

    const char* initializers[] = {"K-Means++", "Random"};
    int currentInit = (pendingConfig.init == common::InitializationType::KMEANS_PLUSPLUS) ? 0 : 1;
    if (ImGui::Combo("Initialization", &currentInit, initializers, 2)) {
        pendingConfig.init =
            (currentInit == 0) ? common::InitializationType::KMEANS_PLUSPLUS : common::InitializationType::RANDOM;
        configChanged = true;
    }

    const char* engines[] = {"Classical (CPU)", "Quantum"};
    int currentEngine = (pendingConfig.algorithm == common::AlgorithmType::KMEANS_REGULAR) ? 0 : 1;
    if (ImGui::Combo("Execution Engine", &currentEngine, engines, 2)) {
        pendingConfig.algorithm =
            (currentEngine == 0) ? common::AlgorithmType::KMEANS_REGULAR : common::AlgorithmType::KMEANS_QUANTUM;
        configChanged = true;
    }

    if (configChanged) {
        std::scoped_lock<std::mutex> lock(m_configMutex);
        m_uiConfig = pendingConfig; // flush pending changes atomically
    }

    ImGui::Separator();
    ImGui::Text("Visualization Overlays");
    ImGui::Checkbox("Show Spatial Centroids", &m_showCentroids);
    if (ImGui::Button("Reset Centroids (Flush Memory)")) {
        m_forceReset = true;
    }

    ImGui::Separator();
    ImGui::Separator();
    ImGui::Text("Performance Dashboard");
    
    static float displayUIRenderFPS = 0.0f;
    static auto lastUIUpdateTime = std::chrono::high_resolution_clock::now();
    auto uiNow = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(uiNow - lastUIUpdateTime).count() > 1000) {
        displayUIRenderFPS = ImGui::GetIO().Framerate;
        lastUIUpdateTime = uiNow;
    }
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "UI Render Speed: %.1f FPS", displayUIRenderFPS);

    static uint32_t lastProcessedFrames = 0;
    static std::deque<float> algoFpsHistory;

    const auto [workerFps, algoTimeMs, currentFrames] = [this]() {
        std::scoped_lock<std::mutex> lock(m_dataMutex);
        return std::make_tuple(m_currentWorkerFps, m_currentAlgoTimeMs, m_processedFrames);
    }();

    if (currentFrames != lastProcessedFrames) {
        if (algoFpsHistory.size() >= constants::FPS_HISTORY_WINDOW) {
            algoFpsHistory.pop_front(); // O(1) vs O(n) erase(begin())
        }
        algoFpsHistory.push_back(workerFps);
        lastProcessedFrames = currentFrames;
    }

    if (!algoFpsHistory.empty()) {
        float minFps = algoFpsHistory[0];
        float maxFps = algoFpsHistory[0];
        float sumFps = 0.0f;
        for (float f : algoFpsHistory) {
            minFps = std::min(f, minFps);
            maxFps = std::max(f, maxFps);
            sumFps += f;
        }
        float avgFps = sumFps / static_cast<float>(algoFpsHistory.size());

        // Text Smoothing Timer (0.5s)
        static float displayFps = 0.0f;
        static float displayMs = 0.0f;
        static float displayAvg = 0.0f;
        static float displayMin = 0.0f;
        static float displayMax = 0.0f;
        static auto lastUpdateTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTime).count() > 500) {
            displayFps = workerFps; // Show true instantaneous FPS
            displayMs = algoTimeMs;
            displayAvg = avgFps;
            displayMin = minFps;
            displayMax = maxFps;
            lastUpdateTime = now;
        }

        ImGui::Text("Algorithm Execution: %.2f ms", displayMs);
        ImGui::Text("Camera Pipeline: %.1f FPS", displayFps);
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Avg: %.1f | Min: %.1f | Max: %.1f", displayAvg, displayMin, displayMax);

        // Symmetric Max Filter Graph Smoothing to hide 1-frame calculation drops without tail artifacts
        std::vector<float> fpsPlotBuf;
        int window = 15;
        for (int i = 0; i < static_cast<int>(algoFpsHistory.size()); ++i) {
            float m = 0;
            int start = std::max(0, i - window/2);
            int end = std::min(static_cast<int>(algoFpsHistory.size()) - 1, i + window/2);
            for(int j = start; j <= end; ++j) {
                m = std::max(m, algoFpsHistory[j]);
            }
            fpsPlotBuf.push_back(m);
        }

        ImGui::PlotLines("##Pipeline History", fpsPlotBuf.data(), static_cast<int>(fpsPlotBuf.size()), 0,
                         nullptr, 0.0f, (maxFps * 1.5f) + 5.0f, ImVec2(0, 60));
    }

    ImGui::Separator();
    ImGui::Text("Static Benchmarking");
    if (m_benchmarkState == BenchmarkState::IDLE) {
        if (ImGui::Button("Capture & Run Comparison", ImVec2(-1, 30))) {
            m_benchmarkState = BenchmarkState::CAPTURING;
            m_benchmarkStatusText = "Requesting frame from camera thread...";
        }
    } else if (m_benchmarkState == BenchmarkState::CAPTURING) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "%s", m_benchmarkStatusText.c_str());
    } else if (m_benchmarkState == BenchmarkState::COMPUTING) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.9f, 1.0f), "%s", m_benchmarkStatusText.c_str());
        if (m_benchmarkFuture.valid() && m_benchmarkFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            m_benchmarkResults = m_benchmarkFuture.get();
            m_benchTexturesLoaded = false;
            m_benchmarkState = BenchmarkState::DONE;
        }
    } else if (m_benchmarkState == BenchmarkState::DONE) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Benchmark Complete.");
        ImGui::Text("View the full-screen results.");
    }

    ImGui::End();

    // 2. Video Feed Window
    ImGui::SetNextWindowPos(ImVec2(constants::UI_PANEL_WIDTH, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(
        ImVec2(ImGui::GetIO().DisplaySize.x - constants::UI_PANEL_WIDTH, ImGui::GetIO().DisplaySize.y),
        ImGuiCond_Always);
    ImGui::Begin("Video Segmentation Feed", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    cv::Mat localOriginal;
    cv::Mat localSegmented;
    {
        std::scoped_lock<std::mutex> lock(m_dataMutex);
        if (!m_latestOriginal.empty()) {
            localOriginal = m_latestOriginal.clone();
            localSegmented = m_latestSegmented.clone();
        }
    }

    if (!localOriginal.empty()) {
        matToTexture(localOriginal, m_originalTexture);
        matToTexture(localSegmented, m_segmentedTexture);

        ImGui::BeginGroup();
        ImGui::Text("Original Frame");
        ImVec2 imgSize(static_cast<float>(localOriginal.cols) * 0.8f, static_cast<float>(localOriginal.rows) * 0.8f);
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(m_originalTexture.id)), imgSize, ImVec2(1, 0),
                     ImVec2(0, 1));
        ImGui::EndGroup();

        ImGui::SameLine();

        ImGui::BeginGroup();
        ImGui::Text("Clustered Frame");
        ImVec2 segSize(static_cast<float>(localSegmented.cols) * 0.8f, static_cast<float>(localSegmented.rows) * 0.8f);
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(m_segmentedTexture.id)), segSize, ImVec2(1, 0),
                     ImVec2(0, 1));
        ImGui::EndGroup();
    } else {
        ImGui::Text("Warming up camera thread...");
    }

    ImGui::End();

    // 3. Full Screen Benchmark Overlay
    if (m_benchmarkState == BenchmarkState::DONE && m_benchmarkResults) {
        if (!m_benchTexturesLoaded) {
            auto drawCentroids = [](cv::Mat& img, const std::vector<cv::Vec<float, 5>>& centers) {
                for (const auto& c : centers) {
                    cv::Point pt(
                        static_cast<int>((c[3] / constants::SPATIAL_SCALE) * static_cast<float>(img.cols)),
                        static_cast<int>((c[4] / constants::SPATIAL_SCALE) * static_cast<float>(img.rows)));
                    cv::Scalar color(c[0] / constants::COLOR_SCALE, c[1] / constants::COLOR_SCALE, c[2] / constants::COLOR_SCALE);
                    cv::circle(img, pt, 6, color, -1);
                    cv::circle(img, pt, 8, cv::Scalar(255, 255, 255), 2);
                }
            };
            drawCentroids(m_benchmarkResults->classicalSegmented, m_benchmarkResults->classicalCenters);
            drawCentroids(m_benchmarkResults->quantumSegmented, m_benchmarkResults->quantumCenters);
            
            matToTexture(m_benchmarkResults->originalFrame, m_benchOriginalTexture);
            matToTexture(m_benchmarkResults->classicalSegmented, m_benchClassicalTexture);
            matToTexture(m_benchmarkResults->quantumSegmented, m_benchQuantumTexture);
            m_benchTexturesLoaded = true;
        }

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Side-by-Side Algorithm Comparison", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        
        // Calculate metric styles (colors and % differences)
        struct MetricStyle { ImVec4 c1; ImVec4 c2; std::string t1; std::string t2; };
        auto getStyle = [](float v1, float v2, bool lowerIsBetter, bool isInt = false) -> MetricStyle {
            MetricStyle s;
            char b1[64], b2[64];
            if (isInt) { snprintf(b1, sizeof(b1), "%d", (int)v1); snprintf(b2, sizeof(b2), "%d", (int)v2); }
            else if (v1 > 10.0f || v2 > 10.0f) { snprintf(b1, sizeof(b1), "%.2f", v1); snprintf(b2, sizeof(b2), "%.2f", v2); }
            else { snprintf(b1, sizeof(b1), "%.4f", v1); snprintf(b2, sizeof(b2), "%.4f", v2); }
            s.t1 = b1; s.t2 = b2;
            s.c1 = ImVec4(1,1,1,1); s.c2 = ImVec4(1,1,1,1);
            if (std::abs(v1 - v2) > 1e-5f) {
                bool v1Better = lowerIsBetter ? (v1 < v2) : (v1 > v2);
                s.c1 = v1Better ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                s.c2 = !v1Better ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                float worse = v1Better ? v2 : v1;
                float pct = std::abs(worse) > 1e-5f ? (std::abs(v1 - v2) / std::abs(worse)) * 100.0f : 0.0f;
                char pb[32]; snprintf(pb, sizeof(pb), " (%.1f%% better)", pct);
                if (v1Better) s.t1 += pb; else s.t2 += pb;
            }
            return s;
        };

        const auto& cm = m_benchmarkResults->classicalMetrics;
        const auto& qm = m_benchmarkResults->quantumMetrics;
        auto s_wcss = getStyle(cm.wcss, qm.wcss, true);
        auto s_db   = getStyle(cm.daviesBouldin, qm.daviesBouldin, true);
        auto s_sil  = getStyle(cm.silhouetteScore, qm.silhouetteScore, false);
        auto s_iter = getStyle(cm.iterations, qm.iterations, true, true);
        auto s_lat  = getStyle(cm.executionTimeMs, qm.executionTimeMs, true);

        if (ImGui::BeginTable("BenchTable", 3, ImGuiTableFlags_None)) {
            // Calculate dynamic image size to fit columns perfectly
            float colWidth = ImGui::GetWindowWidth() / 3.0f - 15.0f; // add margin
            float ratio = static_cast<float>(m_benchmarkResults->originalFrame.rows) / static_cast<float>(m_benchmarkResults->originalFrame.cols);
            ImVec2 size(colWidth, colWidth * ratio);

            // Row 1: Headers & Images
            ImGui::TableNextRow();
            
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("1. Original Frame");
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(m_benchOriginalTexture.id)), size, ImVec2(1, 0), ImVec2(0, 1));
            
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("2. Classical K-Means");
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(m_benchClassicalTexture.id)), size, ImVec2(1, 0), ImVec2(0, 1));
            
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("3. Quantum K-Means");
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(m_benchQuantumTexture.id)), size, ImVec2(1, 0), ImVec2(0, 1));

            // Row 2: Metrics
            ImGui::TableNextRow();
            
            ImGui::TableSetColumnIndex(0);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Metrics Guide:");
            ImGui::Text("WCSS (Inertia): ↓ Lower is better");
            ImGui::Text("Davies-Bouldin: ↓ Lower is better");
            ImGui::Text("Silhouette: ↑ Higher is better");
            ImGui::Text("Iterations: ↓ Lower is better");
            ImGui::Text("Latency: ↓ Lower is better");
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
        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        
        float btnWidth = 300.0f;
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - btnWidth) * 0.5f);
        if (ImGui::Button("Resume Live Feed", ImVec2(btnWidth, 40))) {
            m_benchmarkState = BenchmarkState::IDLE;
            m_benchmarkResults.reset();
        }
        
        ImGui::End();
    }

    // Rendering commands
    ImGui::Render();
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(m_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::run() {
    m_running = true;
    m_workerThread = std::thread([this]() {
        // Try hardware accelerated capture first (Windows)
        cv::VideoCapture cap(0, cv::CAP_FFMPEG);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
        } else {
            // Fallback
            cap.open(0);
        }

        if (!cap.isOpened()) {
            std::cerr << "Failed to open webcam inside worker thread." << std::endl;

            return;
        }

        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, -6);

        while (m_running) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                continue;
            }

            if (m_benchmarkState == BenchmarkState::CAPTURING) {
                m_benchmarkState = BenchmarkState::COMPUTING;
                m_benchmarkStatusText = "Extracting frame and running dual-engine comparison...";
                cv::Mat benchFrame = frame.clone();
                common::SegmentationConfig benchConfig;
                {
                    std::scoped_lock<std::mutex> lock(m_configMutex);
                    benchConfig = m_uiConfig;
                }
                
                m_benchmarkFuture = std::async(std::launch::async, [benchFrame, benchConfig]() {
                    BenchmarkComparisonResult result;
                    result.originalFrame = benchFrame.clone();
                    
                    auto runEngine = [&](common::AlgorithmType algo, cv::Mat& outSeg, clustering::metrics::BenchmarkResults& outMetrics, std::vector<cv::Vec<float, 5>>& outCenters) {
                        clustering::ClusteringManager mgr;
                        common::SegmentationConfig cfg = benchConfig;
                        cfg.algorithm = algo;
                        mgr.getConfig() = cfg;
                        
                        cv::Mat smallFrame;
                        cv::resize(benchFrame, smallFrame, cv::Size(constants::PROCESS_WIDTH, constants::PROCESS_HEIGHT));
                        
                        auto start = std::chrono::high_resolution_clock::now();
                        cv::Mat segmented = mgr.segmentFrame(smallFrame); 
                        auto end = std::chrono::high_resolution_clock::now();
                        float execMs = std::chrono::duration<float, std::milli>(end - start).count();
                        
                        outCenters = mgr.getCenters();
                        int iterations = mgr.getEngine() ? mgr.getEngine()->getLastIterations() : 0;
                        cv::resize(segmented, outSeg, benchFrame.size(), 0, 0, cv::INTER_NEAREST);
                        
                        int n = smallFrame.rows * smallFrame.cols;
                        cv::Mat samples(n, 5, CV_32F);
                        float colorScale = constants::COLOR_SCALE;
                        float spatialScale = constants::SPATIAL_SCALE;
                        for (int y = 0; y < smallFrame.rows; ++y) {
                            for (int x = 0; x < smallFrame.cols; ++x) {
                                cv::Vec3b px = smallFrame.at<cv::Vec3b>(y, x);
                                int idx = y * smallFrame.cols + x;
                                float* ptr = samples.ptr<float>(idx);
                                ptr[0] = static_cast<float>(px[0]) * colorScale;
                                ptr[1] = static_cast<float>(px[1]) * colorScale;
                                ptr[2] = static_cast<float>(px[2]) * colorScale;
                                ptr[3] = static_cast<float>(x) * spatialScale;
                                ptr[4] = static_cast<float>(y) * spatialScale;
                            }
                        }
                        outMetrics = clustering::metrics::computeAllMetrics(samples, outCenters, iterations, execMs);
                    };
                    
                    runEngine(common::AlgorithmType::KMEANS_REGULAR, result.classicalSegmented, result.classicalMetrics, result.classicalCenters);
                    runEngine(common::AlgorithmType::KMEANS_QUANTUM, result.quantumSegmented, result.quantumMetrics, result.quantumCenters);
                    
                    return result;
                });
            }

            // Pause live tracking if benchmark is active
            if (m_benchmarkState != BenchmarkState::IDLE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Safely ingest latest config from UI panel
            common::SegmentationConfig currentConfig;
            {
                std::scoped_lock<std::mutex> lock(m_configMutex);
                currentConfig = m_uiConfig;
            }
            m_manager.getConfig() = currentConfig;

            if (m_forceReset) {
                m_manager.resetCenters();
                m_forceReset = false;
            }

            cv::Mat smallFrame;
            cv::resize(frame, smallFrame, cv::Size(constants::PROCESS_WIDTH, constants::PROCESS_HEIGHT));

            auto startAlgo = std::chrono::high_resolution_clock::now();
            cv::Mat segmented = m_manager.segmentFrame(smallFrame); // Returned by value, clone() not needed
            auto endAlgo = std::chrono::high_resolution_clock::now();

            float algoTimeMs = std::chrono::duration<float, std::milli>(endAlgo - startAlgo).count();
            float pipelineFps = 0.0f;
            if (m_lastWorkerTime.time_since_epoch().count() != 0) {
                float elapsed = std::chrono::duration<float>(endAlgo - m_lastWorkerTime).count();
                pipelineFps = 1.0f / elapsed;
            }
            m_lastWorkerTime = endAlgo;

            const auto& centers = m_manager.getCenters();

            cv::Mat displaySegmented;
            cv::resize(segmented, displaySegmented, frame.size(), 0, 0, cv::INTER_NEAREST);

            if (m_showCentroids) {
                for (const auto& c : centers) {
                    cv::Point centerPt(
                        static_cast<int>((c[3] / constants::SPATIAL_SCALE) * static_cast<float>(frame.cols)),
                        static_cast<int>((c[4] / constants::SPATIAL_SCALE) * static_cast<float>(frame.rows)));
                    cv::Scalar color(c[0] / constants::COLOR_SCALE, c[1] / constants::COLOR_SCALE,
                                     c[2] / constants::COLOR_SCALE);
                    cv::circle(displaySegmented, centerPt, 6, color, -1);
                    cv::circle(displaySegmented, centerPt, 8, cv::Scalar(255, 255, 255), 2);
                }
            }

            // Push results safely
            {
                std::scoped_lock<std::mutex> lock(m_dataMutex);
                m_latestOriginal = std::move(frame);
                m_latestSegmented = std::move(displaySegmented);
                m_latestCenters = centers;
                m_currentWorkerFps = pipelineFps;
                m_currentAlgoTimeMs = algoTimeMs;
                m_processedFrames++;
            }
        }
    });

    // Natively uncapped ImGui loop
    while (glfwWindowShouldClose(m_window) == 0) {
        glfwPollEvents();
        renderUI();
        glfwSwapBuffers(m_window);
    }

    m_running = false;
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
}

} // namespace kmeans::io
