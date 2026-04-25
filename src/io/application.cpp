#include "io/application.hpp"

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

    ImGui::StyleColorsDark();

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

void Application::matToTexture(const cv::Mat& mat, TextureResource& textureRes) {
    if (mat.empty()) {
        return;
    }

    cv::Mat rgbMat;
    cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGBA);

    if (textureRes.id == 0) {
        glGenTextures(1, &textureRes.id);
    }

    glBindTexture(GL_TEXTURE_2D, textureRes.id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

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

    ImGui::Text("Core Hyperparameters");
    configChanged |= ImGui::SliderInt("Clusters (k)", &m_uiConfig.k, constants::K_MIN, constants::K_MAX);
    configChanged |= ImGui::SliderInt("Learning Interval", &m_uiConfig.learningInterval, constants::LEARN_INTERVAL_MIN,
                                      constants::LEARN_INTERVAL_MAX);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "How many frames to cache clusters before re-running K-Means. Set to 1 to force calculation every frame.");
    }

    ImGui::Separator();

    ImGui::Text("Architecture Strategy");

    const char* preprocessors[] = {"Full Data (Flatten)", "RCC Tree (Coreset)"};
    int currentPreprocessor = (m_uiConfig.strategy == common::DataStrategy::FULL_DATA) ? 0 : 1;
    if (ImGui::Combo("Data Preprocessor", &currentPreprocessor, preprocessors, 2)) {
        m_uiConfig.strategy =
            (currentPreprocessor == 0) ? common::DataStrategy::FULL_DATA : common::DataStrategy::RCC_TREES;
        configChanged = true;
    }

    const char* initializers[] = {"K-Means++", "Random"};
    int currentInit = (m_uiConfig.init == common::InitializationType::KMEANS_PLUSPLUS) ? 0 : 1;
    if (ImGui::Combo("Initialization", &currentInit, initializers, 2)) {
        m_uiConfig.init =
            (currentInit == 0) ? common::InitializationType::KMEANS_PLUSPLUS : common::InitializationType::RANDOM;
        configChanged = true;
    }

    const char* engines[] = {"Classical (CPU)", "Quantum"};
    int currentEngine = (m_uiConfig.algorithm == common::AlgorithmType::KMEANS_REGULAR) ? 0 : 1;
    if (ImGui::Combo("Execution Engine", &currentEngine, engines, 2)) {
        m_uiConfig.algorithm =
            (currentEngine == 0) ? common::AlgorithmType::KMEANS_REGULAR : common::AlgorithmType::KMEANS_QUANTUM;
        configChanged = true;
    }

    if (configChanged) {
        std::scoped_lock<std::mutex> lock(m_configMutex);
    }

    ImGui::Separator();
    ImGui::Text("Visualization Overlays");
    ImGui::Checkbox("Show Spatial Centroids", &m_showCentroids);
    if (ImGui::Button("Reset Centroids (Flush Memory)")) {
        m_forceReset = true;
    }

    ImGui::Separator();

    ImGui::Text("Performance Dashboard");
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "UI Render Speed: %.1f FPS", ImGui::GetIO().Framerate);

    static uint32_t lastProcessedFrames = 0;
    static std::vector<float> algoFpsHistory;

    const auto [workerFps, algoTimeMs, currentFrames] = [this]() {
        std::scoped_lock<std::mutex> lock(m_dataMutex);
        return std::make_tuple(m_currentWorkerFps, m_currentAlgoTimeMs, m_processedFrames);
    }();

    if (currentFrames != lastProcessedFrames) {
        if (algoFpsHistory.size() >= constants::FPS_HISTORY_WINDOW) {
            algoFpsHistory.erase(algoFpsHistory.begin());
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

        ImGui::Text("Algorithm Execution: %.2f ms", algoTimeMs);
        ImGui::Text("Camera Pipeline: %.1f FPS", workerFps);
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Avg: %.1f | Min: %.1f | Max: %.1f", avgFps, minFps, maxFps);

        ImGui::PlotLines("##Pipeline History", algoFpsHistory.data(), static_cast<int>(algoFpsHistory.size()), 0,
                         nullptr, 0.0f, (maxFps * 1.5f) + 5.0f, ImVec2(0, 80));
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
        cv::VideoCapture cap(0, cv::CAP_MSMF);
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

        while (m_running) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
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
