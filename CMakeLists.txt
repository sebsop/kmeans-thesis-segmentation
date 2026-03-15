cmake_minimum_required(VERSION 3.18)
project(KMeansThesisSegmentation LANGUAGES CXX CUDA)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

enable_language(CUDA)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Core Logic Library
add_library(kmeans_core STATIC
    src/video_io.cpp
    src/coreset.cpp
    src/rcc.cpp
    src/utils.cpp
    src/clustering/clustering_entry.cpp
    src/clustering/kmeans_regular.cpp  
)

set_source_files_properties(src/BACKEND/CudaKernels.cu PROPERTIES LANGUAGE CUDA)
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