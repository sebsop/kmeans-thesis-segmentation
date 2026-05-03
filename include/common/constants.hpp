#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace kmeans::constants {

// API Flags & Hardware
constinit inline const int VIZ_RESIZE_ALGO = cv::INTER_NEAREST;
constinit inline const int CAMERA_HW_ACCEL = cv::VIDEO_ACCELERATION_ANY;
constinit inline const int VIZ_OUTLINE_COLOR = 255;

// Layout Geometry
constinit inline const int UI_FPS_PLOT_WINDOW = 15;
constinit inline const int UI_BENCH_COL_COUNT = 3;
constinit inline const float UI_BENCH_BTN_PADDING = 20.0f;
constinit inline const float UI_BENCH_SLIDER_SPACING = 30.0f;

// Quantum Range
constinit inline const float QUANTUM_RANGE_EPSILON = 1e-8f;

// Color structures for logic encapsulation
struct ColorRGBA {
    float r, g, b, a;
};

// Premium Visual Identity
constexpr ColorRGBA UI_COLOR_BG_DARK = {.r = 0.06f, .g = 0.06f, .b = 0.07f, .a = 1.0f};
constexpr ColorRGBA UI_COLOR_ACCENT = {.r = 0.6f, .g = 0.4f, .b = 0.9f, .a = 1.0f};
constexpr ColorRGBA UI_COLOR_TEXT_DIM = {.r = 0.5f, .g = 0.5f, .b = 0.5f, .a = 1.0f};

// UITheme Palettes
namespace theme {
constexpr ColorRGBA TEXT = {.r = 1.00f, .g = 1.00f, .b = 1.00f, .a = 1.00f};
constexpr ColorRGBA WINDOW_BG = {.r = 0.08f, .g = 0.08f, .b = 0.08f, .a = 0.94f};
constexpr ColorRGBA POPUP_BG = {.r = 0.12f, .g = 0.12f, .b = 0.12f, .a = 0.94f};
constexpr ColorRGBA BORDER = {.r = 0.43f, .g = 0.43f, .b = 0.50f, .a = 0.50f};
constexpr ColorRGBA FRAME_BG = {.r = 0.16f, .g = 0.16f, .b = 0.18f, .a = 1.00f};
constexpr ColorRGBA FRAME_BG_HOVERED = {.r = 0.24f, .g = 0.24f, .b = 0.28f, .a = 1.00f};
constexpr ColorRGBA FRAME_BG_ACTIVE = {.r = 0.35f, .g = 0.35f, .b = 0.40f, .a = 1.00f};
constexpr ColorRGBA TITLE_BG = {.r = 0.08f, .g = 0.08f, .b = 0.08f, .a = 1.00f};
constexpr ColorRGBA TITLE_BG_ACTIVE = {.r = 0.08f, .g = 0.08f, .b = 0.08f, .a = 1.00f};
constexpr ColorRGBA BUTTON = {.r = 0.24f, .g = 0.24f, .b = 0.28f, .a = 1.00f};
constexpr ColorRGBA BUTTON_HOVERED = {.r = 0.35f, .g = 0.35f, .b = 0.40f, .a = 1.00f};
constexpr ColorRGBA BUTTON_ACTIVE = {.r = 0.45f, .g = 0.45f, .b = 0.50f, .a = 1.00f};
constexpr ColorRGBA HEADER = {.r = 0.24f, .g = 0.24f, .b = 0.28f, .a = 1.00f};
constexpr ColorRGBA HEADER_HOVERED = {.r = 0.35f, .g = 0.35f, .b = 0.40f, .a = 1.00f};
constexpr ColorRGBA HEADER_ACTIVE = {.r = 0.45f, .g = 0.45f, .b = 0.50f, .a = 1.00f};
constexpr ColorRGBA CHECK_MARK = {.r = 0.60f, .g = 0.40f, .b = 0.90f, .a = 1.00f};
constexpr ColorRGBA SLIDER_GRAB = {.r = 0.60f, .g = 0.40f, .b = 0.90f, .a = 1.00f};
constexpr ColorRGBA SLIDER_GRAB_ACTIVE = {.r = 0.70f, .g = 0.50f, .b = 1.00f, .a = 1.00f};
constexpr ColorRGBA PLOT_LINES = {.r = 0.60f, .g = 0.40f, .b = 0.90f, .a = 1.00f};
constexpr ColorRGBA PLOT_LINES_HOVERED = {.r = 0.70f, .g = 0.50f, .b = 1.00f, .a = 1.00f};
constexpr ColorRGBA SUCCESS_COL = {.r = 0.40f, .g = 1.00f, .b = 0.40f, .a = 1.00f};
constexpr ColorRGBA ERROR_COL = {.r = 1.00f, .g = 0.40f, .b = 0.40f, .a = 1.00f};
constexpr ColorRGBA WARNING_COL = {.r = 0.80f, .g = 0.80f, .b = 0.20f, .a = 1.00f};

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

// Logic Defaults
constexpr int DEFAULT_STRIDE = 4;
constexpr int DEFAULT_LEARN_INTERVAL = 15;
constexpr int STABLE_RANDOM_SEED = 42;

// Layout & Ratios
constexpr float UI_LANDING_OFFSET = 15.0f;
constexpr float UI_K_SLIDER_WIDTH = 120.0f;
constexpr float UI_STRIDE_SLIDER_WIDTH = 100.0f;
constexpr float UI_RADIO_WIDTH = 260.0f;

// Scientific Accuracy & Metrics
constexpr int FEATURE_DIMS = 5;
constexpr float COLOR_MAX_F = 255.0f;
constexpr int METRIC_APPROX_SUBSET_SIZE = 2000;
constexpr float MATH_EPSILON = 1e-6f;
constexpr float MATH_INF = 1e30f;

// Quantum Kernel Math
constexpr float QUANTUM_PHASE_OFFSET = 0.5f;

// UI Layout Polish
constexpr float UI_BTN_WIDTH_LG = 250.0f;
constexpr float UI_BTN_WIDTH_MD = 150.0f;
constexpr float UI_BTN_HEIGHT = 40.0f;
constexpr float UI_BORDER_THICKNESS = 5.0f;
constexpr int UI_LAYOUT_GAPS = 12;
constexpr float UI_LAYOUT_PADDING = 10.0f;
constexpr int UI_REFRESH_FAST = 500;
constexpr int UI_REFRESH_SLOW = 1000;
constexpr float UI_BENCH_IMG_SCALE = 0.825f;
constexpr double UI_ANIM_DOT_SPEED = 4.0;

// Threading & OS
constexpr int WORKER_FPS_MATCH_SLEEP = 33;

// Internal pipeline sizes
constexpr int VIDEO_WIDTH = 640;
constexpr int VIDEO_HEIGHT = 480;
constexpr int PROCESS_WIDTH = 320;
constexpr int PROCESS_HEIGHT = 240;
constexpr int FPS_HISTORY_WINDOW = 90;

// Mathematics & Scaling
constexpr float PI_F = 3.1415926535f;
constexpr float COLOR_SCALE = 1.0f / 255.0f;
constexpr float SPATIAL_SCALE = 1.0f / (static_cast<float>(PROCESS_WIDTH) * static_cast<float>(PROCESS_HEIGHT));

// Algorithm & Benchmark Convergence
constexpr int K_MIN = 2;
constexpr int K_MAX = 20;
constexpr int LEARN_INTERVAL_MIN = 1;
constexpr int LEARN_INTERVAL_MAX = 60;
constexpr int SAMPLE_COUNT = 5000;
constexpr int BENCHMARK_MAX_ITERATIONS = 1000;
constexpr float CONVERGENCE_EPSILON = 1e-4f;

// Hardware & CUDA Configuration
constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr int CUDA_BLOCK_2D_X = 16;
constexpr int CUDA_BLOCK_2D_Y = 16;
constexpr int CAMERA_AUTO_EXPOSURE = -6;

// Visualization & Drawing
constexpr int VIZ_CENTROID_RADIUS = 6;
constexpr int VIZ_OUTLINE_WIDTH = 8;
constexpr int WINDOW_WIDTH = 1750;
constexpr int WINDOW_HEIGHT = 700;

// UI Layout boundaries
constexpr float UI_PANEL_WIDTH = 400.0f;
constexpr float UI_PLOT_HEIGHT = 60.0f;
constexpr float UI_WINDOW_PADDING = 10.0f;

} // namespace kmeans::constants

namespace kmeans {
using FeatureVector = cv::Vec<float, constants::FEATURE_DIMS>;
}