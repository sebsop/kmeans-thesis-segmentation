#include "io/application.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Backend configurations
#include "common/enums.hpp"
#include "common/config.hpp"

// Windows <GL/gl.h> strictly supports OpenGL 1.1. 
// GL_CLAMP_TO_EDGE (0x812F) was introduced in OpenGL 1.2, so we manually define the constant
// to avoid importing bulky extension wranglers (GLEW/GLAD) just for a single texture wrap setting.
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace kmeans {
namespace io {

    Application::Application() {
        initWindow();
        initImGui();
        m_uiConfig = m_manager.getConfig();
        m_initialized = true;
    }

    Application::~Application() {
        if (m_initialized) {
            cleanup();
        }
    }

    void Application::initWindow() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW. Application cannot start." << std::endl;
            exit(EXIT_FAILURE);
        }

        // GL 3.0 + GLSL 130
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        m_window = glfwCreateWindow(1280, 720, "K-Means Segmentation Thesis - ImGui Dashboard", NULL, NULL);
        if (m_window == NULL) {
            std::cerr << "Failed to create window using GLFW." << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(m_window);
        glfwSwapInterval(0); // Disable vsync to uncap from monitor refresh rate
    }

    void Application::initImGui() {
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        // Setup Dear ImGui style (Dark modern look)
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(m_window, true);
        ImGui_ImplOpenGL3_Init("#version 130");
    }

    void Application::cleanup() {
        // Cleanup textures
        if (m_originalTexture) glDeleteTextures(1, &m_originalTexture);
        if (m_segmentedTexture) glDeleteTextures(1, &m_segmentedTexture);

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    // Helper method to convert an OpenCV Mat (BGR format) to an OpenGL Texture
    GLuint Application::matToTexture(const cv::Mat& mat, GLuint existingTexture) {
        if (mat.empty()) return existingTexture;

        cv::Mat rgbMat;
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGBA);

        GLuint textureId = existingTexture;
        if (textureId == 0) {
            glGenTextures(1, &textureId);
        }

        glBindTexture(GL_TEXTURE_2D, textureId);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Upload pixels into texture
        // Avoid glPixelStorei(GL_UNPACK_ALIGNMENT, 1); since OpenCV usually aligns to 4 bytes for RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgbMat.cols, rgbMat.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgbMat.ptr());

        return textureId;
    }

    void Application::renderUI() {
        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Control Panel
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(350, ImGui::GetIO().DisplaySize.y), ImGuiCond_Always);
        ImGui::Begin("Clustering Controls", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
        
        // Grab local config lock only when combo boxes change
        bool configChanged = false;
        
        ImGui::Text("Core Hyperparameters");
        configChanged |= ImGui::SliderInt("Clusters (k)", &m_uiConfig.k, 2, 20);
        configChanged |= ImGui::SliderInt("Learning Interval", &m_uiConfig.learningInterval, 1, 60);
        if (ImGui::IsItemHovered()) {
             ImGui::SetTooltip("How many frames to cache clusters before re-running K-Means. Set to 1 to force calculation every frame.");
        }
        
        ImGui::Separator();
        
        ImGui::Text("Architecture Strategy");
        
        const char* preprocessors[] = { "Full Data (Flatten)", "RCC Tree (Coreset)" };
        int currentPreprocessor = (m_uiConfig.strategy == DataStrategy::FULL_DATA) ? 0 : 1;
        if (ImGui::Combo("Data Preprocessor", &currentPreprocessor, preprocessors, 2)) {
            m_uiConfig.strategy = (currentPreprocessor == 0) ? DataStrategy::FULL_DATA : DataStrategy::RCC_TREES;
            configChanged = true;
        }

        const char* initializers[] = { "K-Means++", "Random" };
        int currentInit = (m_uiConfig.init == InitializationType::KMEANS_PLUSPLUS) ? 0 : 1;
        if (ImGui::Combo("Initialization", &currentInit, initializers, 2)) {
            m_uiConfig.init = (currentInit == 0) ? InitializationType::KMEANS_PLUSPLUS : InitializationType::RANDOM;
            configChanged = true;
        }

        const char* engines[] = { "Classical (CPU)", "Quantum" };
        int currentEngine = (m_uiConfig.algorithm == AlgorithmType::KMEANS_REGULAR) ? 0 : 1;
        if (ImGui::Combo("Execution Engine", &currentEngine, engines, 2)) {
             m_uiConfig.algorithm = (currentEngine == 0) ? AlgorithmType::KMEANS_REGULAR : AlgorithmType::KMEANS_QUANTUM;
             configChanged = true;
        }
        
        // Save config thread-safely
        if (configChanged) {
            std::lock_guard<std::mutex> lock(m_configMutex);
        }
        
        ImGui::Separator();
        ImGui::Text("Visualization Overlays");
        ImGui::Checkbox("Show Spatial Centroids", &m_showCentroids);
        
        ImGui::Separator();
        
        // Advanced metrics like FPS graph
        ImGui::Text("Performance Dashboard");
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "UI Render Speed: %.1f FPS", ImGui::GetIO().Framerate);
        
        static uint32_t lastProcessedFrames = 0;
        static std::vector<float> algoFpsHistory;
        
        float workerFps = 0.0f;
        float algoTimeMs = 0.0f;
        uint32_t currentFrames = 0;
        {
            std::lock_guard<std::mutex> lock(m_dataMutex);
            workerFps = m_currentWorkerFps;
            algoTimeMs = m_currentAlgoTimeMs;
            currentFrames = m_processedFrames;
        }

        if (currentFrames != lastProcessedFrames) {
            if (algoFpsHistory.size() >= 90) { // Keep last 90 frames (~3 sec at 30 fps)
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
                if (f < minFps) minFps = f;
                if (f > maxFps) maxFps = f;
                sumFps += f;
            }
            float avgFps = sumFps / algoFpsHistory.size();

            ImGui::Text("Algorithm Execution: %.2f ms", algoTimeMs);
            ImGui::Text("Camera Pipeline: %.1f FPS", workerFps);
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Avg: %.1f | Min: %.1f | Max: %.1f", avgFps, minFps, maxFps);
            
            // Render the graph line for the background pipeline speed!
            ImGui::PlotLines("##Pipeline History", algoFpsHistory.data(), static_cast<int>(algoFpsHistory.size()), 0, NULL, 0.0f, maxFps * 1.5f + 5.0f, ImVec2(0, 80));
        }

        ImGui::End();

        // 2. Video Feed Window
        ImGui::SetNextWindowPos(ImVec2(350, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x - 350, ImGui::GetIO().DisplaySize.y), ImGuiCond_Always);
        ImGui::Begin("Video Segmentation Feed", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
        
        cv::Mat localOriginal;
        cv::Mat localSegmented;
        {
            std::lock_guard<std::mutex> lock(m_dataMutex);
            if (!m_latestOriginal.empty()) {
                localOriginal = m_latestOriginal.clone();
                localSegmented = m_latestSegmented.clone();
            }
        }
        
        if (!localOriginal.empty()) {
            m_originalTexture = matToTexture(localOriginal, m_originalTexture);
            m_segmentedTexture = matToTexture(localSegmented, m_segmentedTexture);

            ImGui::Text("Original Frame");
            ImVec2 imgSize(localOriginal.cols * 0.5f, localOriginal.rows * 0.5f);
            ImGui::Image((void*)(intptr_t)m_originalTexture, imgSize);
            
            ImGui::SameLine();

            ImGui::Text("Clustered Frame");
            ImVec2 segSize(localSegmented.cols * 0.5f, localSegmented.rows * 0.5f);
            ImGui::Image((void*)(intptr_t)m_segmentedTexture, segSize);
        } else {
            ImGui::Text("Warming up camera thread...");
        }

        ImGui::End();

        // Rendering commands
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(m_window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void Application::run() {
        m_running = true;
        m_workerThread = std::thread([this]() {
            cv::VideoCapture cap(0);
            if (!cap.isOpened()) {
                std::cerr << "Failed to open webcam inside worker thread." << std::endl;
                return;
            }

            while (m_running) {
                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) continue;

                // Safely ingest latest config from UI panel
                SegmentationConfig currentConfig;
                {
                    std::lock_guard<std::mutex> lock(m_configMutex);
                    currentConfig = m_uiConfig;
                }
                m_manager.getConfig() = currentConfig;

                cv::Mat smallFrame;
                cv::resize(frame, smallFrame, cv::Size(200, 150)); 
                
                auto startAlgo = std::chrono::high_resolution_clock::now();
                cv::Mat segmented = m_manager.segmentFrame(smallFrame).clone();
                auto endAlgo = std::chrono::high_resolution_clock::now();
                
                float algoTimeMs = std::chrono::duration<float, std::milli>(endAlgo - startAlgo).count();
                float pipelineFps = 0.0f;
                if (m_lastWorkerTime.time_since_epoch().count() != 0) {
                    float elapsed = std::chrono::duration<float>(endAlgo - m_lastWorkerTime).count();
                    pipelineFps = 1.0f / elapsed;
                }
                m_lastWorkerTime = endAlgo;

                auto centers = m_manager.getCenters();

                cv::Mat displaySegmented;
                cv::resize(segmented, displaySegmented, frame.size(), 0, 0, cv::INTER_NEAREST);

                if (m_showCentroids) {
                    float scaleX = (float)frame.cols / smallFrame.cols;
                    float scaleY = (float)frame.rows / smallFrame.rows;
                    for (const auto& c : centers) {
                        cv::Point centerPt(static_cast<int>(c[3] * scaleX), static_cast<int>(c[4] * scaleY));
                        cv::Scalar color(c[0], c[1], c[2]);
                        cv::circle(displaySegmented, centerPt, 12, color, -1);
                        cv::circle(displaySegmented, centerPt, 14, cv::Scalar(255, 255, 255), 2);
                    }
                }

                // Push results safely
                {
                    std::lock_guard<std::mutex> lock(m_dataMutex);
                    m_latestOriginal = frame;
                    m_latestSegmented = displaySegmented;
                    m_latestCenters = centers;
                    m_currentWorkerFps = pipelineFps;
                    m_currentAlgoTimeMs = algoTimeMs;
                    m_processedFrames++;
                }
            }
        });

        // Natively uncapped ImGui loop
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            renderUI();
            glfwSwapBuffers(m_window);
        }

        m_running = false;
        if (m_workerThread.joinable()) {
            m_workerThread.join();
        }
    }

}
}
