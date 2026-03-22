cmake_minimum_required(VERSION 3.18)
project(KMeansThesisSegmentation LANGUAGES CXX CUDA)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Core Logic Library
add_library(kmeans_core
    src/backend/cuda_kernels.cu
    src/clustering/classical_engine.cpp
    src/clustering/clustering_manager.cpp
    src/clustering/clustering_factory.cpp
    src/clustering/full_data_preprocessor.cpp
    src/clustering/rcc_data_preprocessor.cpp
    src/clustering/random_initializer.cpp
    src/clustering/kmeans_plus_plus_initializer.cpp
    src/core/coreset.cpp
    src/core/rcc.cpp
    src/common/utils.cpp
    src/io/video_io.cpp
)

set_source_files_properties(src/backend/cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(kmeans_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON       # Enable separable compilation for CUDA (different from .cpp files)
    CUDA_STANDARD 14                    # Use C++14 standard for CUDA files
    CUDA_STANDARD_REQUIRED ON           # Enforce the use of the specified C++ standard
    CUDA_ARCHITECTURES "native"         # Specify target architecture (native for the current machine)
)

target_link_libraries(kmeans_core PUBLIC ${OpenCV_LIBS})

# Main Executable
add_executable(kmeans_thesis_segmentation src/main.cpp)

# Set CUDA properties for the main executable
set_target_properties(kmeans_thesis_segmentation PROPERTIES CUDA_ARCHITECTURES "native")

target_link_libraries(kmeans_thesis_segmentation PRIVATE kmeans_core)