#pragma once

namespace kmeans::constants {

// Color structures for logic encapsulation
struct ColorRGBA {
    float r, g, b, a;
};

// Premium Visual Identity
constexpr ColorRGBA UI_COLOR_BG_DARK = {0.06f, 0.06f, 0.07f, 1.0f};
constexpr ColorRGBA UI_COLOR_ACCENT = {0.6f, 0.4f, 0.9f, 1.0f};
constexpr ColorRGBA UI_COLOR_TEXT_DIM = {0.5f, 0.5f, 0.5f, 1.0f};

// UITheme Palettes
struct UITheme {
    static constexpr ColorRGBA TEXT = {1.00f, 1.00f, 1.00f, 1.00f};
    static constexpr ColorRGBA WINDOW_BG = {0.08f, 0.08f, 0.08f, 0.94f};
    static constexpr ColorRGBA POPUP_BG = {0.12f, 0.12f, 0.12f, 0.94f};
    static constexpr ColorRGBA BORDER = {0.43f, 0.43f, 0.50f, 0.50f};
    static constexpr ColorRGBA FRAME_BG = {0.16f, 0.16f, 0.18f, 1.00f};
    static constexpr ColorRGBA FRAME_BG_HOVERED = {0.24f, 0.24f, 0.28f, 1.00f};
    static constexpr ColorRGBA FRAME_BG_ACTIVE = {0.35f, 0.35f, 0.40f, 1.00f};
    static constexpr ColorRGBA TITLE_BG = {0.08f, 0.08f, 0.08f, 1.00f};
    static constexpr ColorRGBA TITLE_BG_ACTIVE = {0.08f, 0.08f, 0.08f, 1.00f};
    static constexpr ColorRGBA BUTTON = {0.24f, 0.24f, 0.28f, 1.00f};
    static constexpr ColorRGBA BUTTON_HOVERED = {0.35f, 0.35f, 0.40f, 1.00f};
    static constexpr ColorRGBA BUTTON_ACTIVE = {0.45f, 0.45f, 0.50f, 1.00f};
    static constexpr ColorRGBA HEADER = {0.24f, 0.24f, 0.28f, 1.00f};
    static constexpr ColorRGBA HEADER_HOVERED = {0.35f, 0.35f, 0.40f, 1.00f};
    static constexpr ColorRGBA HEADER_ACTIVE = {0.45f, 0.45f, 0.50f, 1.00f};
    static constexpr ColorRGBA CHECK_MARK = {0.60f, 0.40f, 0.90f, 1.00f};
    static constexpr ColorRGBA SLIDER_GRAB = {0.60f, 0.40f, 0.90f, 1.00f};
    static constexpr ColorRGBA SLIDER_GRAB_ACTIVE = {0.70f, 0.50f, 1.00f, 1.00f};
    static constexpr ColorRGBA PLOT_LINES = {0.60f, 0.40f, 0.90f, 1.00f};
    static constexpr ColorRGBA PLOT_LINES_HOVERED = {0.70f, 0.50f, 1.00f, 1.00f};

    static constexpr float WINDOW_ROUNDING = 8.0f;
    static constexpr float CHILD_ROUNDING = 6.0f;
    static constexpr float FRAME_ROUNDING = 6.0f;
    static constexpr float POPUP_ROUNDING = 6.0f;
    static constexpr float GRAB_ROUNDING = 6.0f;
    static constexpr float FRAME_PADDING_X = 10.0f;
    static constexpr float FRAME_PADDING_Y = 6.0f;
    static constexpr float WINDOW_PADDING_X = 12.0f;
    static constexpr float WINDOW_PADDING_Y = 12.0f;
    static constexpr float ITEM_SPACING_X = 8.0f;
    static constexpr float ITEM_SPACING_Y = 8.0f;
};

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

// Mathematics & Scaling
constexpr float PI_F = 3.1415926535f;
constexpr float COLOR_SCALE = 1.0f / 255.0f;
constexpr float SPATIAL_SCALE = 1.0f / 76800.0f;

// Algorithm & Benchmark Convergence
constexpr int K_MIN = 2;
constexpr int K_MAX = 20;
constexpr int LEARN_INTERVAL_MIN = 1;
constexpr int LEARN_INTERVAL_MAX = 60;
constexpr int SAMPLE_COUNT = 5000;
constexpr int BENCHMARK_MAX_ITERATIONS = 1000;
constexpr float CONVERGENCE_EPSILON = 1e-4f;

// Internal pipeline sizes
constexpr int VIDEO_WIDTH = 640;
constexpr int VIDEO_HEIGHT = 480;
constexpr int PROCESS_WIDTH = 320;
constexpr int PROCESS_HEIGHT = 240;
constexpr int FPS_HISTORY_WINDOW = 90;

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