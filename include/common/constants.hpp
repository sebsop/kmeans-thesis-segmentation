#pragma once

namespace kmeans::constants {

constexpr int VIDEO_WIDTH = 640;
constexpr int VIDEO_HEIGHT = 480;
constexpr float COLOR_SCALE = 1.0f;
constexpr float SPATIAL_SCALE = 10.0f;

constexpr int K_MIN = 2;
constexpr int K_MAX = 20;
constexpr int LEARN_INTERVAL_MIN = 1;
constexpr int LEARN_INTERVAL_MAX = 60;
constexpr int SAMPLE_COUNT = 2000;

// Internal pipeline sizes
constexpr int PROCESS_WIDTH = 640;
constexpr int PROCESS_HEIGHT = 480;
constexpr int FPS_HISTORY_WINDOW = 90;

// UI Layout boundaries
constexpr float UI_PANEL_WIDTH = 400.0f;

} // namespace kmeans::constants