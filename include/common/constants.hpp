#pragma once

namespace kmeans::constants {

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