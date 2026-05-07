# Real-Time Hybrid Quantum-Classical K-Means Segmentation

A high-performance computer vision system for real-time image segmentation, featuring a custom hybrid architecture that bridges classical GPU acceleration (CUDA) with quantum-inspired distance metrics.

## Academic Context

This project was developed as the core practical component for my **Bachelor's Thesis at Babeș-Bolyai University**. 

Having started my journey into Quantum Computing during my second year of study and subsequently completing various projects in the field, integrating quantum principles into my final thesis was a personal "must-have." This project represents a convergence of my interests in high-performance computing, artificial intelligence, and quantum mechanics, serving as a foundation for my future research goals in **Quantum Machine Learning (QML)**.

The fundamental architecture evolved from a previous academic project completed during my *Parallel and Distributed Programming* course: [Real-Time Parallel K-Means Image Segmentation](https://github.com/sebsop/realtime-parallel-kmeans-segmentation). While that work focused on distributed CPU parallelization (MPI/OpenMP), this thesis shifts the focus toward maximizing single-node throughput via low-level CUDA optimization and the exploration of non-Euclidean quantum-inspired similarity metrics.

## Overview

This repository implements a highly modular, professional-grade C++/CUDA pipeline capable of segmenting live video feeds at high frame rates. It compares traditional Euclidean distance clustering against a novel, quantum-inspired phase-estimation metric (simulated via GPU).

### Key Features

*   **Hybrid Engine Architecture**: Hot-swap between a highly optimized Classical CUDA engine and a Quantum-inspired emulation engine in real-time.
*   **Quantum-Inspired Metric**: Implements a simulated Swap-Test interference approximation to calculate vector similarity using Hilbert-space phase overlap rather than standard Euclidean distance.
*   **High-Performance CUDA Backend**: Custom CUDA kernels for spatial preprocessing, K-Means++ initialization, and massive parallel pixel assignment utilizing shared memory optimization.
*   **Scientific Benchmarking**: Integrated real-time metric calculation including approximated Silhouette Scores, Davies-Bouldin Index, and Within-Cluster Sum of Squares (WCSS).
*   **Modern UI Integration**: Decoupled, thread-safe Dear ImGui interface providing dynamic parameter control, live telemetry, and side-by-side visual comparison.
*   **Audit-Ready Codebase**: Comprehensive Doxygen documentation, strict C++17 adherence, RAII resource management, and a robust GoogleTest verification suite.

## Architecture Highlights

The system is designed with strict separation of concerns, utilizing modern software engineering patterns:

*   **Factory Pattern**: Dynamically instantiates the correct execution engine and initializer based on runtime configuration.
*   **Observer Pattern**: Ensures the high-frequency CUDA processing loop remains completely decoupled from the UI rendering thread.
*   **Temporal Coherence Optimization**: Implements configurable centroid memoization (learning intervals) to drastically reduce GPU computational load during stable video scenes.

## Getting Started

### Prerequisites

*   C++17 compatible compiler (MSVC / GCC / Clang)
*   CMake 3.20+
*   NVIDIA CUDA Toolkit 12.x+
*   OpenCV 4.x
*   GoogleTest (Fetched automatically via CMake)
*   Dear ImGui (Included/Fetched via build system)

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/sebsop/kmeans-thesis-segmentation.git
   cd kmeans-thesis-segmentation
   ```
2. Configure and build via CMake:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```
3. Run the executable generated in your build directory.

## Repository Structure

```text
kmeans-thesis-segmentation
├─ .clang-format
├─ .clang-tidy
├─ assets
├─ CMakeLists.txt
├─ CMakeSettings.json
├─ include
│  ├─ backend
│  │  └─ cuda_assignment_context.hpp
│  ├─ clustering
│  │  ├─ clustering_factory.hpp
│  │  ├─ clustering_manager.hpp
│  │  ├─ engines
│  │  │  ├─ base_kmeans_engine.hpp
│  │  │  ├─ classical_engine.hpp
│  │  │  ├─ kmeans_engine.hpp
│  │  │  └─ quantum_engine.hpp
│  │  ├─ initializers
│  │  │  ├─ initializer.hpp
│  │  │  ├─ kmeans_plus_plus_initializer.hpp
│  │  │  └─ random_initializer.hpp
│  │  ├─ metrics.hpp
│  │  └─ preprocessor
│  │     └─ strided_data_preprocessor.hpp
│  ├─ common
│  │  ├─ config.hpp
│  │  ├─ constants.hpp
│  │  ├─ enums.hpp
│  │  ├─ utils.hpp
│  │  └─ vector_math.hpp
│  └─ io
│     ├─ application.hpp
│     ├─ benchmark_command.hpp
│     ├─ benchmark_observer.hpp
│     ├─ benchmark_result.hpp
│     ├─ benchmark_runner.hpp
│     ├─ ui
│     │  ├─ benchmark_overlay_ui.hpp
│     │  ├─ control_panel_ui.hpp
│     │  └─ video_feed_ui.hpp
│     └─ ui_manager.hpp
├─ LICENSE
├─ README.md
├─ src
│  ├─ backend
│  │  └─ cuda_kernels.cu
│  ├─ clustering
│  │  ├─ clustering_factory.cpp
│  │  ├─ clustering_manager.cpp
│  │  ├─ engines
│  │  │  ├─ base_kmeans_engine.cu
│  │  │  ├─ classical_engine.cu
│  │  │  └─ quantum_engine.cu
│  │  ├─ initializers
│  │  │  ├─ kmeans_plus_plus_initializer.cu
│  │  │  └─ random_initializer.cpp
│  │  ├─ metrics.cpp
│  │  └─ preprocessor
│  │     └─ strided_data_preprocessor.cu
│  ├─ io
│  │  ├─ application.cpp
│  │  ├─ benchmark_command.cpp
│  │  ├─ benchmark_runner.cpp
│  │  ├─ ui
│  │  │  ├─ benchmark_overlay_ui.cpp
│  │  │  ├─ control_panel_ui.cpp
│  │  │  └─ video_feed_ui.cpp
│  │  └─ ui_manager.cpp
│  ├─ main.cpp
│  └─ vendor/                    # Third-party dependencies (ImGui, etc.)
└─ tests
   ├─ backend
   │  └─ cuda_kernels_tests.cu
   ├─ clustering
   │  ├─ clustering_factory_tests.cpp
   │  ├─ clustering_manager_tests.cpp
   │  ├─ engines
   │  │  ├─ base_kmeans_engine_tests.cu
   │  │  ├─ classical_engine_tests.cu
   │  │  └─ quantum_engine_tests.cu
   │  ├─ initializers
   │  │  ├─ kmeans_plus_plus_initializer_tests.cu
   │  │  └─ random_initializer_tests.cu
   │  ├─ metrics_tests.cpp
   │  └─ preprocessor
   │     └─ strided_data_preprocessor_tests.cu
   ├─ CMakeLists.txt
   ├─ common
   │  └─ vector_math_tests.cpp
   └─ io
      └─ ui_integration_tests.cpp
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Sebastian Soptelea**  
Babeș-Bolyai University  
Email: [sebastian.soptelea@proton.me](mailto:sebastian.soptelea@proton.me)  
GitHub: [@sebsop](https://github.com/sebsop)