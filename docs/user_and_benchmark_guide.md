# User & Benchmark Guide: Real-Time Quantum-Inspired 5D K-Means Segmentation

This guide explains how to operate the application, tune clustering parameters in real-time, run built-in diagnostics, and execute scientific benchmarks.

---

## 1. Interactive GUI Dashboard

The GUI is built using Dear ImGui and GLFW, providing a dashboard split into a control panel, live telemetry plots, and side-by-side video feeds.

### Parameter Tuning

You can adjust the following parameters on the fly in the control panel:

- **Stride**: Pixel downsampling step size (range: `1` to `16`).
  - *Setting 1*: Processes every single pixel. Extremely detailed but computationally demanding.
  - *Setting 16*: Samples every 16th pixel. Provides a massive speedup, suitable for lower-spec GPUs.
- **K (Clusters)**: The number of distinct segmented colors to generate (range: `2` to `40`).
- **Learning Interval**: The frame interval between K-Means centroid updates (range: `1` to `60`).
  - A higher interval stops re-clustering on every frame, reducing GPU utilization when the camera is stable. Set to `1` to force calculation on every frame.

### Centroid Visualizations & Reset

- **Show Spatial Centroids**: Toggles a visual overlay that projects the 5D centroids (using color BGR and mapping the XY spatial coordinates to the display resolution) and draws circles with white borders directly over the video frames.
- **Reset Centroids (Flush Memory)**: Flushes the memory of current centroids to force K-Means re-initialization.

### Engine Hot-Swapping

In the UI, you can select:
- **Classical Engine**: Uses the standard Euclidean distance metric.
- **Quantum-Inspired Engine**: Emulates a quantum SWAP test, mapping BGR-XY vectors to angular phase states in Hilbert space.

Transitions between engines are completely seamless and execute without interrupting the video feed.

---

## 2. Telemetry & Scientific Diagnostics

The UI thread tracks algorithm and pipeline performance metrics dynamically.

### Live Telemetry Metrics
- **Core K-Means Latency**: Time spent executing the selected clustering engine (Classical Lloyd's loop or Quantum-inspired phase-mapping) on the GPU (updates only on active learning frames).
- **Total Frame Latency**: Total time spent on GPU/CPU preprocessing, K-Means calculation (if active), and pixel assignment for the current frame.
- **Raw Engine Speed**: The maximum throughput of the core clustering engine alone if it executed on every single frame without temporal caching (1000ms / Core K-Means Latency).
- **Actual Frame Rate**: The active processing frame rate of the worker thread, capped by the camera's hardware capture rate.
- **Overall Throughput**: Rolling average of the actual overall system processing throughput (1000ms / Average Frame Latency) computed over a window of active and cached frames.

---

## 3. Still-Frame Capture & Comparative Benchmarks

The system includes a premium comparative benchmarking engine designed for scientific analysis.

### Still-Frame Comparison Mode
By clicking the **"Capture & Run Comparison"** button, the system freezes the current video feed and performs an off-line benchmark. It displays three side-by-side streams:
1. **Original Frame** (Non-segmented, raw input)
2. **Classical K-Means** (Segmented using the Classical engine)
3. **Quantum K-Means** (Segmented using the Quantum-Inspired engine)

### Scientific Scorecard Table
The comparison screen displays a detailed performance scorecard comparing the two backends across five metrics:
*   **WCSS (Inertia)** (Compactness of clusters; lower is better)
*   **Davies-Bouldin Index** (Cluster separation; lower is better)
*   **Approx Silhouette Score** (Cluster assignment quality; higher is better. Approximated via a Monte Carlo simulation of $2000$ points to remain fast)
*   **Iteration Count** (Number of iterations taken to converge; lower is better)
*   **Latency** (Core execution time in milliseconds; lower is better)

The scorecard dynamically highlights the superior engine for each metric in **green** with the calculated percentage improvement, and the slower engine in **red**.

### Still-Frame Parameter Tuning & Rerunning
While in the comparison view, you can interactive tweak the parameters for the captured frame:
- Change **K** (Clusters) from `2` to `40`.
- Change **Stride** from `1` to `16`.
- Select the **Initialization Strategy** (K-Means++ vs. Random).
- Toggle **Show Centroids** to display the calculated centroids on the captured frame.

Modifying any of these parameters or clicking **"Rerun Frame"** immediately re-executes both engines on the exact same still frame, allowing real-time scientific comparison of the algorithms. Click **"Resume Live Feed"** to return to real-time video processing.

# 4. Building and Running the Application

### Prerequisites
Make sure you have installed:
*   C++20 compatible compiler (MSVC, GCC, or Clang)
*   CMake 3.18+
*   NVIDIA CUDA Toolkit 12.x+
*   OpenCV 4.x
*   GLFW3

### Build Instructions
Run the standard CMake compilation sequence:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Launching the Application
Once the build is complete, launch the generated graphical executable:
```bash
# Windows (from the build directory)
./Release/kmeans_thesis_segmentation.exe
```
