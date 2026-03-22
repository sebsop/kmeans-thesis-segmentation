#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <thread>
#include <mutex>
#include <atomic>

// Include GLFW wrapper
#include <GLFW/glfw3.h>

#include "clustering/clustering_manager.hpp"
#include <chrono>

namespace kmeans {
namespace io {

    class Application {
    private:
        GLFWwindow* m_window = nullptr;
        clustering::ClusteringManager m_manager;

        // OpenGL Textures for rendering OpenCV matrices in ImGui
        GLuint m_originalTexture = 0;
        GLuint m_segmentedTexture = 0;

        // Multithreading Synchronization
        std::thread m_workerThread;
        std::atomic<bool> m_running{false};
        std::mutex m_dataMutex;
        std::mutex m_configMutex;
        
        cv::Mat m_latestOriginal;
        cv::Mat m_latestSegmented;
        std::vector<cv::Vec<float, 5>> m_latestCenters;
        SegmentationConfig m_uiConfig;
        
        std::chrono::high_resolution_clock::time_point m_lastWorkerTime;
        float m_currentWorkerFps = 0.0f;
        float m_currentAlgoTimeMs = 0.0f;
        uint32_t m_processedFrames = 0;

        // State initialized flags
        bool m_initialized = false;
        bool m_showCentroids = false;

        void initWindow();
        void initImGui();
        void cleanup();

        void renderUI();
        GLuint matToTexture(const cv::Mat& mat, GLuint existingTexture);

    public:
        Application();
        ~Application();

        void run();
    };

}
}
