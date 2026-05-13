/**
 * @file application.cpp
 * @brief Core application lifecycle and thread orchestration.
 */

#include "io/application.hpp"

#include <chrono>
#include <cuda_runtime.h>
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

/**
 * @brief Initializes the application context, window, and UI systems.
 */
Application::Application() : m_uiConfig(m_manager.getConfig()) {
    initWindow();
    initImGui();
    UIManager::applyPremiumTheme();
    // Connect the UI to benchmark updates
    m_benchmarkRunner.addObserver(&m_uiManager);
}

/**
 * @brief Ensures clean shutdown of threads and resources.
 */
Application::~Application() noexcept {
    m_benchmarkRunner.removeObserver(&m_uiManager);
    cleanup();
}

/**
 * @brief Creates the GLFW window and applies platform-specific styling.
 */
void Application::initWindow() {
    if (glfwInit() == 0) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(constants::ui::WINDOW_WIDTH, constants::ui::WINDOW_HEIGHT,
                                "K-Means Real-Time Segmentation Benchmark", nullptr, nullptr);
    if (m_window == nullptr) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

#if defined(_WIN32)
    // Windows-specific: Load application icon from assets
    HWND hwnd = glfwGetWin32Window(m_window);
    HINSTANCE hInstance = GetModuleHandle(nullptr);

    HICON hIcon = nullptr;
    const char* iconPaths[] = {"assets/icon.ico", "../assets/icon.ico", "../../assets/icon.ico",
                               "../../../assets/icon.ico"};

    [[maybe_unused]] auto* it = std::ranges::find_if(iconPaths, [&](const char* path) {
        hIcon =
            reinterpret_cast<HICON>(LoadImageA(hInstance, path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE));
        return hIcon != nullptr;
    });

    if (hIcon != nullptr) {
        SendMessage(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hIcon));
        SendMessage(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hIcon));
    }
#endif

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0); // VSync disabled to measure uncapped performance
}

/**
 * @brief Configures the ImGui context for OpenGL3.
 */
void Application::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

/**
 * @brief Stops the worker thread and destroys graphics contexts.
 */
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

/**
 * @brief Entry point for the dual-threaded execution model.
 *
 * 1. Spawns a worker thread for Camera I/O and CUDA processing.
 * 2. Runs the main UI loop on the primary thread.
 */
void Application::run() {
    m_running = true;

    // --- WORKER THREAD (Camera & CUDA) ---
    m_workerThread = std::thread([this]() {
        cv::VideoCapture cap(0, cv::CAP_FFMPEG);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_HW_ACCELERATION, constants::video::HW_ACCEL);
        } else {
            cap.open(0);
        }

        if (!cap.isOpened()) {
            std::cerr << "Critical: Failed to open webcam." << std::endl;
            return;
        }

        // Verify CUDA availability before starting the heavy pipeline
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
            std::cerr << "Critical: No CUDA-capable GPU detected.\n";
            m_running = false;
            return;
        }

        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, constants::video::CAMERA_AUTO_EXPOSURE);

        while (m_running) [[likely]] {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                continue;
            }

            // Handle Asynchronous Benchmark Requests
            auto bState = m_benchmarkRunner.getState();
            if (bState == BenchmarkState::CAPTURING || bState == BenchmarkState::RECOMPUTING) {
                common::SegmentationConfig benchConfig;
                {
                    std::scoped_lock<std::mutex> lock(m_configMutex);
                    benchConfig = m_uiConfig;
                }
                m_benchmarkRunner.startComputing(frame, benchConfig);
            }

            // Pause live background processing if a benchmark is active
            bState = m_benchmarkRunner.getState();
            if (bState != BenchmarkState::IDLE) {
                std::scoped_lock<std::mutex> lock(m_dataMutex);
                m_initialized = true;
                continue;
            }

            // Normal Real-Time Pipeline
            cv::Mat processFrame;
            cv::resize(frame, processFrame,
                       cv::Size(constants::video::PROCESS_WIDTH, constants::video::PROCESS_HEIGHT));

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

            // Results preparation
            cv::Mat segmentedFull;
            cv::resize(segmented, segmentedFull, frame.size(), 0, 0, constants::viz::RESIZE_ALGO);

            std::vector<FeatureVector> centers;
            if (m_showCentroids) {
                centers = m_manager.getCenters();
                // Draw centroids directly on the frame for low-latency visual feedback
                std::ranges::for_each(centers, [&](const auto& c) {
                    cv::Point pt(
                        static_cast<int>((c[3] / constants::video::SPATIAL_WEIGHT) * static_cast<float>(frame.cols)),
                        static_cast<int>((c[4] / constants::video::SPATIAL_WEIGHT) * static_cast<float>(frame.rows)));
                    cv::Scalar color(c[0] / constants::video::COLOR_SCALE, c[1] / constants::video::COLOR_SCALE,
                                     c[2] / constants::video::COLOR_SCALE);
                    cv::circle(segmentedFull, pt, constants::viz::CENTROID_RADIUS, color, -1);
                    cv::circle(segmentedFull, pt, constants::viz::OUTLINE_WIDTH,
                               cv::Scalar(constants::viz::OUTLINE_COLOR, constants::viz::OUTLINE_COLOR,
                                          constants::viz::OUTLINE_COLOR),
                               2);
                });
            }

            auto now = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(now - m_lastWorkerTime).count();
            float instFps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
            m_lastWorkerTime = now;

            // Commit results to shared context
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

    // --- MAIN THREAD (UI & Rendering) ---
    while (glfwWindowShouldClose(m_window) == 0 && m_running) [[likely]] {
        glfwPollEvents();

        // Manage UI configuration persistence during benchmarking
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

        // Show loading screen until the worker thread provides the first frame
        if (!m_initialized) {
            UIManager::renderLoadingScreen(m_window);
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

        UIDataContext ctx{.latestOriginal = displayOriginal,
                          .latestSegmented = displaySegmented,
                          .uiConfig = m_uiConfig,
                          .configMutex = m_configMutex,
                          .showCentroids = m_showCentroids,
                          .forceReset = m_forceReset,
                          .currentWorkerFps = m_currentWorkerFps,
                          .currentAlgoTimeMs = m_currentAlgoTimeMs,
                          .processedFrames = m_processedFrames,
                          .benchmarkRunner = m_benchmarkRunner};

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
