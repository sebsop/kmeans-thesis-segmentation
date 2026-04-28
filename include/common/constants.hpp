#pragma once

namespace kmeans::constants {

constexpr int VIDEO_WIDTH = 640;
constexpr int VIDEO_HEIGHT = 480;
constexpr float COLOR_SCALE = 1.0f / 255.0f;
constexpr float SPATIAL_SCALE = 1.0f / (76800.0f * 2.0f);

constexpr int K_MIN = 2;
constexpr int K_MAX = 20;
constexpr int LEARN_INTERVAL_MIN = 1;
constexpr int LEARN_INTERVAL_MAX = 60;
constexpr int SAMPLE_COUNT = 5000;

// Internal pipeline sizes
constexpr int PROCESS_WIDTH = 320;
constexpr int PROCESS_HEIGHT = 240;
constexpr int FPS_HISTORY_WINDOW = 90;

// UI Layout boundaries
constexpr float UI_PANEL_WIDTH = 400.0f;

} // namespace kmeans::constants