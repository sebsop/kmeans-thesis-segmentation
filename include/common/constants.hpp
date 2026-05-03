#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace kmeans::constants {

// Color structures for logic encapsulation
struct ColorRGBA {
    float r, g, b, a;
};

namespace math {
    constexpr float PI_F = 3.1415926535f;
    constexpr float EPSILON = 1e-6f;
    constexpr float INF = 1e30f;
} // namespace math

namespace clustering {
    constexpr int FEATURE_DIMS = 5;
    constexpr int K_MIN = 2;
    constexpr int K_MAX = 20;
    constexpr int LEARN_INTERVAL_MIN = 1;
    constexpr int LEARN_INTERVAL_MAX = 60;
    constexpr int DEFAULT_STRIDE = 4;
    constexpr int DEFAULT_LEARN_INTERVAL = 15;
    constexpr int SAMPLE_COUNT = 5000;
    constexpr int BENCHMARK_MAX_ITERATIONS = 1000;
    constexpr float CONVERGENCE_EPSILON = 1e-4f;
    constexpr int STABLE_RANDOM_SEED = 42;
} // namespace clustering

namespace video {
    constexpr int WIDTH = 640;
    constexpr int HEIGHT = 480;
    constexpr int PROCESS_WIDTH = 320;
    constexpr int PROCESS_HEIGHT = 240;
    constexpr float COLOR_SCALE = 1.0f / 255.0f;
    constexpr float SPATIAL_SCALE = 1.0f / (320.0f * 240.0f);

    constexpr int CAMERA_AUTO_EXPOSURE = -6;
    constinit inline const int HW_ACCEL = cv::VIDEO_ACCELERATION_ANY;
} // namespace video

namespace quantum {
    constexpr float PHASE_OFFSET = 0.5f;
    constexpr float RANGE_EPSILON = 1e-8f;
    constexpr float SCALE_FACTOR = math::PI_F / 2.0f;
} // namespace quantum

namespace cuda {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int BLOCK_2D_X = 16;
    constexpr int BLOCK_2D_Y = 16;
} // namespace cuda

namespace ui {
    constexpr int WINDOW_WIDTH = 1750;
    constexpr int WINDOW_HEIGHT = 700;
    constexpr float PANEL_WIDTH = 400.0f;
    constexpr float PLOT_HEIGHT = 60.0f;
    constexpr float WINDOW_PADDING = 10.0f;

    constexpr int FPS_PLOT_WINDOW = 15;
    constexpr int BENCH_COL_COUNT = 3;
    constexpr float BENCH_BTN_PADDING = 20.0f;
    constexpr float BENCH_SLIDER_SPACING = 30.0f;

    constexpr float LANDING_OFFSET = 15.0f;
    constexpr float K_SLIDER_WIDTH = 120.0f;
    constexpr float STRIDE_SLIDER_WIDTH = 100.0f;
    constexpr float RADIO_WIDTH = 260.0f;

    constexpr float BTN_WIDTH_LG = 250.0f;
    constexpr float BTN_WIDTH_MD = 150.0f;
    constexpr float BTN_HEIGHT = 40.0f;
    constexpr float BORDER_THICKNESS = 5.0f;
    constexpr int LAYOUT_GAPS = 12;
    constexpr float LAYOUT_PADDING = 10.0f;
    constexpr int REFRESH_FAST = 500;
    constexpr int REFRESH_SLOW = 1000;
    constexpr float BENCH_IMG_SCALE = 0.825f;
    constexpr double ANIM_DOT_SPEED = 4.0;
    constexpr int FPS_HISTORY_WINDOW = 90;

    namespace theme {
        constexpr ColorRGBA BG_LIGHT = {.r = 0.98f, .g = 0.98f, .b = 0.98f, .a = 1.0f};
        constexpr ColorRGBA ACCENT = {.r = 0.35f, .g = 0.20f, .b = 0.70f, .a = 1.0f};
        constexpr ColorRGBA TEXT_DIM = {.r = 0.45f, .g = 0.45f, .b = 0.45f, .a = 1.0f};

        constexpr ColorRGBA TEXT = {.r = 0.12f, .g = 0.12f, .b = 0.12f, .a = 1.00f};
        constexpr ColorRGBA WINDOW_BG = {.r = 0.96f, .g = 0.96f, .b = 0.97f, .a = 0.98f};
        constexpr ColorRGBA POPUP_BG = {.r = 1.00f, .g = 1.00f, .b = 1.00f, .a = 0.98f};
        constexpr ColorRGBA BORDER = {.r = 0.85f, .g = 0.85f, .b = 0.85f, .a = 1.00f};
        constexpr ColorRGBA FRAME_BG = {.r = 0.90f, .g = 0.90f, .b = 0.92f, .a = 1.00f};
        constexpr ColorRGBA FRAME_BG_HOVERED = {.r = 0.85f, .g = 0.85f, .b = 0.88f, .a = 1.00f};
        constexpr ColorRGBA FRAME_BG_ACTIVE = {.r = 0.80f, .g = 0.80f, .b = 0.84f, .a = 1.00f};
        constexpr ColorRGBA TITLE_BG = {.r = 0.94f, .g = 0.94f, .b = 0.95f, .a = 1.00f};
        constexpr ColorRGBA TITLE_BG_ACTIVE = {.r = 0.90f, .g = 0.90f, .b = 0.92f, .a = 1.00f};
        constexpr ColorRGBA BUTTON = {.r = 0.88f, .g = 0.88f, .b = 0.90f, .a = 1.00f};
        constexpr ColorRGBA BUTTON_HOVERED = {.r = 0.82f, .g = 0.82f, .b = 0.85f, .a = 1.00f};
        constexpr ColorRGBA BUTTON_ACTIVE = {.r = 0.75f, .g = 0.75f, .b = 0.80f, .a = 1.00f};
        constexpr ColorRGBA HEADER = {.r = 0.92f, .g = 0.92f, .b = 0.94f, .a = 1.00f};
        constexpr ColorRGBA HEADER_HOVERED = {.r = 0.88f, .g = 0.88f, .b = 0.90f, .a = 1.00f};
        constexpr ColorRGBA HEADER_ACTIVE = {.r = 0.84f, .g = 0.84f, .b = 0.86f, .a = 1.00f};
        constexpr ColorRGBA CHECK_MARK = {.r = 0.35f, .g = 0.20f, .b = 0.70f, .a = 1.00f};
        constexpr ColorRGBA SLIDER_GRAB = {.r = 0.35f, .g = 0.20f, .b = 0.70f, .a = 1.00f};
        constexpr ColorRGBA SLIDER_GRAB_ACTIVE = {.r = 0.45f, .g = 0.30f, .b = 0.80f, .a = 1.00f};
        constexpr ColorRGBA PLOT_LINES = {.r = 0.35f, .g = 0.20f, .b = 0.70f, .a = 1.00f};
        constexpr ColorRGBA PLOT_LINES_HOVERED = {.r = 0.45f, .g = 0.30f, .b = 0.80f, .a = 1.00f};
        constexpr ColorRGBA SUCCESS_COL = {.r = 0.10f, .g = 0.60f, .b = 0.10f, .a = 1.00f};
        constexpr ColorRGBA ERROR_COL = {.r = 0.80f, .g = 0.10f, .b = 0.10f, .a = 1.00f};
        constexpr ColorRGBA WARNING_COL = {.r = 0.60f, .g = 0.50f, .b = 0.00f, .a = 1.00f};
        constexpr ColorRGBA NEUTRAL_COL = {.r = 0.12f, .g = 0.12f, .b = 0.12f, .a = 1.00f};

        constexpr ColorRGBA BENCH_TITLE_CLASSICAL = {.r = 0.10f, .g = 0.50f, .b = 0.10f, .a = 1.00f};
        constexpr ColorRGBA BENCH_TITLE_QUANTUM = {.r = 0.50f, .g = 0.20f, .b = 0.70f, .a = 1.00f};
        constexpr ColorRGBA BENCH_GUIDE = {.r = 0.40f, .g = 0.40f, .b = 0.40f, .a = 1.00f};

        constexpr float WINDOW_ROUNDING = 8.0f;
        constexpr float CHILD_ROUNDING = 6.0f;
        constexpr float FRAME_ROUNDING = 6.0f;
        constexpr float POPUP_ROUNDING = 6.0f;
        constexpr float GRAB_ROUNDING = 6.0f;
        constexpr float FRAME_PADDING_X = 10.0f;
        constexpr float FRAME_PADDING_Y = 6.0f;
        constexpr float WINDOW_PADDING_X = 12.0f;
        constexpr float WINDOW_PADDING_Y = 12.0f;
        constexpr float ITEM_SPACING_X = 8.0f;
        constexpr float ITEM_SPACING_Y = 8.0f;
    } // namespace theme
} // namespace ui

namespace viz {
    constexpr int CENTROID_RADIUS = 6;
    constexpr int OUTLINE_WIDTH = 8;
    constinit inline const int RESIZE_ALGO = cv::INTER_NEAREST;
    constinit inline const int OUTLINE_COLOR = 255;
} // namespace viz

namespace metrics {
    constexpr int APPROX_SUBSET_SIZE = 2000;
}

namespace threading {
    constexpr int WORKER_FPS_MATCH_SLEEP = 33;
}

} // namespace kmeans::constants

namespace kmeans {
    using FeatureVector = cv::Vec<float, constants::clustering::FEATURE_DIMS>;
}