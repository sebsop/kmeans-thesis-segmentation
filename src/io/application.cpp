#include "io/application.hpp"

#include <chrono>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Platform specific includes for icon loading
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <windows.h>

#include <GLFW/glfw3native.h>
#endif

#include "common/constants.hpp"

namespace kmeans::io {

Application::Application() {
    initWindow();
    initImGui();
    UIManager::applyPremiumTheme();
    m_uiConfig = m_manager.getConfig();
}

Application::~Application() noexcept {
    cleanup();
}

void Application::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(constants::WINDOW_WIDTH, constants::WINDOW_HEIGHT,
                                "K-Means Real-Time Segmentation Benchmark", nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

#if defined(_WIN32)
    HWND hwnd = glfwGetWin32Window(m_window);
    HINSTANCE hInstance = GetModuleHandle(nullptr);

    HICON hIcon = nullptr;
    const char* iconPaths[] = {"assets/icon.ico", "../assets/icon.ico", "../../assets/icon.ico",
                               "../../../assets/icon.ico"};

    for (const char* path : iconPaths) {
        hIcon =
            reinterpret_cast<HICON>(LoadImageA(hInstance, path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE));
        if (hIcon)
            break;
    }

    if (hIcon) {
        SendMessage(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hIcon));
        SendMessage(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hIcon));
    }
#endif

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0); // VSync disabled to measure uncapped UI frame rate
}

void Application::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

void Application::cleanup() noexcept {
    m_running = false;
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Application::run() {
    m_running = true;
    m_workerThread = std::thread([this]() {
        cv::VideoCapture cap(0, cv::CAP_FFMPEG);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
        } else {
            cap.open(0);
        }

        if (!cap.isOpened()) {
            std::cerr << "Failed to open webcam inside worker thread." << std::endl;
            return;
        }

        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, constants::CAMERA_AUTO_EXPOSURE);

        while (m_running) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                continue;
            }

            auto bState = m_benchmarkRunner.getState();
            if (bState == BenchmarkState::CAPTURING || bState == BenchmarkState::RECOMPUTING) {
                common::SegmentationConfig benchConfig;
                {
                    std::scoped_lock<std::mutex> lock(m_configMutex);
                    benchConfig = m_uiConfig;
                }
                m_benchmarkRunner.startComputing(frame, benchConfig);
            }

            // If a benchmark is active, do not run the normal segmentation pipeline in the background.
            // This prevents race conditions and ensures UI parameter changes only affect the benchmark.
            bState = m_benchmarkRunner.getState();
            if (bState != BenchmarkState::IDLE) {
                std::scoped_lock<std::mutex> lock(m_dataMutex);
                m_initialized = true;
                continue;
            }

            cv::Mat processFrame;
            cv::resize(frame, processFrame, cv::Size(constants::PROCESS_WIDTH, constants::PROCESS_HEIGHT));

            {
                std::scoped_lock<std::mutex> lock(m_configMutex);
                m_manager.getConfig() = m_uiConfig;
                if (m_forceReset) {
                    m_manager.resetCenters();
                    m_forceReset = false;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat segmented = m_manager.segmentFrame(processFrame);
            auto end = std::chrono::high_resolution_clock::now();
            float execMs = std::chrono::duration<float, std::milli>(end - start).count();

            cv::Mat segmentedFull;
            cv::resize(segmented, segmentedFull, frame.size(), 0, 0, cv::INTER_NEAREST);

            std::vector<cv::Vec<float, 5>> centers;
            if (m_showCentroids) {
                centers = m_manager.getCenters();
                for (const auto& c : centers) {
                    cv::Point pt(static_cast<int>((c[3] / constants::SPATIAL_SCALE) * static_cast<float>(frame.cols)),
                                 static_cast<int>((c[4] / constants::SPATIAL_SCALE) * static_cast<float>(frame.rows)));
                    cv::Scalar color(c[0] / constants::COLOR_SCALE, c[1] / constants::COLOR_SCALE,
                                     c[2] / constants::COLOR_SCALE);
                    cv::circle(segmentedFull, pt, constants::VIZ_CENTROID_RADIUS, color, -1);
                    cv::circle(segmentedFull, pt, constants::VIZ_OUTLINE_WIDTH, cv::Scalar(255, 255, 255), 2);
                }
            }

            auto now = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(now - m_lastWorkerTime).count();
            float instFps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
            m_lastWorkerTime = now;

            {
                std::scoped_lock<std::mutex> lock(m_dataMutex);
                m_latestOriginal = frame;
                m_latestSegmented = segmentedFull;
                m_latestCenters = centers;
                m_currentWorkerFps = instFps;
                m_currentAlgoTimeMs = execMs;
                m_processedFrames++;
                m_initialized = true;
            }
        }
    });

    while (!glfwWindowShouldClose(m_window) && m_running) {
        glfwPollEvents();

        auto bState = m_benchmarkRunner.getState();
        if (bState != BenchmarkState::IDLE && !m_benchWasActive) {
            std::scoped_lock<std::mutex> lock(m_configMutex);
            m_savedBenchConfig = m_uiConfig;
            m_benchWasActive = true;
        } else if (bState == BenchmarkState::IDLE && m_benchWasActive) {
            std::scoped_lock<std::mutex> lock(m_configMutex);
            m_uiConfig = m_savedBenchConfig;
            m_benchWasActive = false;
        }

        if (!m_initialized) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            int display_w = 0, display_h = 0;
            glfwGetFramebufferSize(m_window, &display_w, &display_h);

            // Premium full-screen loading overlay
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(static_cast<float>(display_w), static_cast<float>(display_h)));

            ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoBackground;

            ImGui::Begin("Loader", nullptr, flags);

            const char* loadingText = "INITIALIZING K-MEANS ENGINE";
            ImVec2 textSize = ImGui::CalcTextSize(loadingText);

            // Animate dots based on time
            int dots = static_cast<int>(ImGui::GetTime() * 4.0) % 4;
            std::string dotStr(dots, '.');
            std::string fullText = std::string(loadingText) + dotStr;

            // Center position for main text
            ImGui::SetCursorPos(ImVec2((display_w - textSize.x) * 0.5f, (display_h - textSize.y) * 0.5f));

            // Use a nice accent color (cyan/blue)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(constants::UI_COLOR_ACCENT.r, constants::UI_COLOR_ACCENT.g,
                                                        constants::UI_COLOR_ACCENT.b, constants::UI_COLOR_ACCENT.a));
            ImGui::TextUnformatted(fullText.c_str());
            ImGui::PopStyleColor();

            // Subtitle
            const char* subText = "Connecting to camera stream and allocating VRAM...";
            ImVec2 subSize = ImGui::CalcTextSize(subText);
            ImGui::SetCursorPos(ImVec2((display_w - subSize.x) * 0.5f, (display_h + textSize.y) * 0.5f + 15.0f));
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(constants::UI_COLOR_TEXT_DIM.r, constants::UI_COLOR_TEXT_DIM.g,
                                         constants::UI_COLOR_TEXT_DIM.b, constants::UI_COLOR_TEXT_DIM.a));
            ImGui::TextUnformatted(subText);
            ImGui::PopStyleColor();

            ImGui::End();

            ImGui::Render();
            glViewport(0, 0, display_w, display_h);

            // Deep dark premium background
            glClearColor(constants::UI_COLOR_BG_DARK.r, constants::UI_COLOR_BG_DARK.g, constants::UI_COLOR_BG_DARK.b,
                         constants::UI_COLOR_BG_DARK.a);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(m_window);
            continue;
        }

        cv::Mat displayOriginal;
        cv::Mat displaySegmented;
        {
            std::scoped_lock<std::mutex> lock(m_dataMutex);
            displayOriginal = m_latestOriginal;
            displaySegmented = m_latestSegmented;
        }

        UIDataContext ctx{displayOriginal,   displaySegmented, m_uiConfig,         m_configMutex,
                          m_showCentroids,   m_forceReset,     m_currentWorkerFps, m_currentAlgoTimeMs,
                          m_processedFrames, m_benchmarkRunner};

        m_uiManager.render(ctx);

        int display_w = 0;
        int display_h = 0;
        glfwGetFramebufferSize(m_window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window);
    }
}

} // namespace kmeans::io
