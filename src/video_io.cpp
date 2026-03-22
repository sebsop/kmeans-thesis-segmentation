#include "video_io.hpp"
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "common/enums.hpp"
#include "common/constants.hpp"
#include "clustering/ClusteringManager.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#ifdef _WIN32 // If on Windows, include Windows.h for window style manipulation
#include <windows.h>
#endif

namespace kmeans {
    // Display webcam feed with real-time k-means segmentation and an adjustable 'K' parameter slider
    void showWebcamFeed()
    {
        cv::VideoCapture cap;
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT);

        unsigned char* pinned_frame_ptr;
        int imgSize = VIDEO_WIDTH * VIDEO_HEIGHT * 3;

        cudaHostAlloc(&pinned_frame_ptr, imgSize, cudaHostAllocDefault);

        cv::Mat frame(VIDEO_WIDTH, VIDEO_HEIGHT, CV_8UC3, pinned_frame_ptr);

        int k_trackbar = 5;

        const std::string windowName = "Realtime Segmentation";

        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("k", windowName, &k_trackbar, K_MAX);
        cv::setTrackbarMin("k", windowName, K_MIN);

#ifdef _WIN32
        // Disable window resizing and maximize button using Windows API
        // (OpenCV does not provide direct functionality for this)
        HWND hwnd = FindWindow(NULL, windowName.c_str()); // Get the window handle

        if (hwnd) {
            LONG style = GetWindowLong(hwnd, GWL_STYLE);
            style &= ~(WS_SIZEBOX | WS_MAXIMIZEBOX);
            style |= WS_BORDER;
            SetWindowLong(hwnd, GWL_STYLE, style);
            SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
#endif

        int64 lastTick = cv::getTickCount();
        double fps = 0.0;
        std::deque<std::pair<double, double>> fpsHistory; // (timestamp, fps)
        double minFps = 0.0, maxFps = 0.0;

        Algorithm algo = Algorithm::KMEANS_REGULAR;
        Algorithm lastAlgo = algo;
        int last_k_trackbar = k_trackbar;

        ClusteringManager manager;
        SegmentationConfig& config = manager.getConfig();

        while (true) {
            cap >> frame;
            if (frame.empty()) break; // Exit if window is closed

            int k = k_trackbar;

            config.algorithm = algo;
            config.k = k;

            cv::Mat localFrame;
            cv::Mat seg;

            seg = manager.segmentFrame(frame);

            if (seg.size() != frame.size()) {
                cv::resize(seg, seg, frame.size(), 0, 0, cv::INTER_NEAREST);
            }

            cv::Mat combined(frame.rows, frame.cols * 2, frame.type());
            frame.copyTo(combined(cv::Rect(0, 0, frame.cols, frame.rows)));
            seg.copyTo(combined(cv::Rect(frame.cols, 0, frame.cols, frame.rows)));

            // FPS tracking
            int64 now = cv::getTickCount();
            double nowSec = now / cv::getTickFrequency();
            double dt = (now - lastTick) / cv::getTickFrequency();
            lastTick = now;
            if (dt > 0) fps = 1.0 / dt;

            // Reset FPS history if backend changed
            if (algo != lastAlgo) {
                fpsHistory.clear();
                minFps = maxFps = fps;

                lastAlgo = algo;
                last_k_trackbar = k_trackbar;
            }

            // Reset FPS history if K changed
            if (k_trackbar != last_k_trackbar) {
                fpsHistory.clear();
                minFps = maxFps = fps;
                lastAlgo = algo;
                last_k_trackbar = k_trackbar;
            }

            // Update FPS history
            fpsHistory.emplace_back(nowSec, fps);

            // Remove old FPS values (older than 3 seconds)
            while (!fpsHistory.empty() && nowSec - fpsHistory.front().first > 3.0)
                fpsHistory.pop_front();

            minFps = maxFps = fps;
            for (const auto& p : fpsHistory) {
                if (p.second < minFps) minFps = p.second;
                if (p.second > maxFps) maxFps = p.second;
            }

            std::string algoName =
                (algo == Algorithm::KMEANS_REGULAR ? "KMEANS" : algo == Algorithm::KMEANS_QUANTUM ? "QUANTUM" : "UNKNOWN");

            std::string overlay = "k=" + std::to_string(k) +
                "  backend=" + algoName +
                "  FPS=" + cv::format("%.1f", fps) +
                "  min=" + cv::format("%.1f", minFps) +
                "  max=" + cv::format("%.1f", maxFps);

            cv::putText(combined, overlay, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(combined, overlay, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            cv::putText(combined, "Original", cv::Point(12, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(combined, "Segmented", cv::Point(frame.cols + 12, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            cv::imshow(windowName, combined);

            char c = (char)cv::waitKey(1);
            if (c == 27) break; // ESC
            if (c == '1') algo = Algorithm::KMEANS_REGULAR;
            if (c == '2') algo = Algorithm::KMEANS_QUANTUM;
        }

        cap.release(); // Clean up camera
        cv::destroyAllWindows(); // Close all OpenCV windows
    }
}