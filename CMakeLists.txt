# =============================================================================
# CMakeLists.txt
# Build system configuration for the K-Means Thesis Pipeline.
#
# This file orchestrates a complex hybrid C++/CUDA project, managing 
# dependencies (OpenCV, ZMQ, GLFW), enforcing code quality standards, 
# and applying high-performance compiler optimizations for real-time 
# computer vision.
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(KMeansThesisSegmentation LANGUAGES CXX CUDA)
 
# --- 1. Modern CMake Policies ---
cmake_policy(SET CMP0135 NEW) # FetchContent timestamp behavior
cmake_policy(SET CMP0146 NEW) # Modern CUDA support

# --- 2. External Dependencies ---
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glfw3 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
# --- 3. Google Test Integration (FetchContent) ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Mark GTest as SYSTEM to suppress warnings from its headers
if(TARGET gtest)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
if(TARGET gtest_main)
    set_target_properties(gtest_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:gtest_main,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
 
enable_testing()
include(CTest)

# --- 4. Build Configurations & Optimizations ---
# Ensure Release build by default to maintain high FPS
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
endif()

# Enable Interprocedural Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Platform-specific compiler optimizations
if(MSVC)
    add_compile_options(
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/fp:fast>" # Prioritize speed over precision for visualization
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/arch:AVX2>"
        # Suppress warnings in external headers
        "$<$<COMPILE_LANGUAGE:CXX>:/external:W0>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>"
        "$<$<COMPILE_LANGUAGE:CXX>:/analyze:external->"
        "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
        "$<$<COMPILE_LANGUAGE:CXX>:/wd6201>"
    )
else()
    add_compile_options(
        "$<$<CONFIG:Release>:-O3>"
        "$<$<CONFIG:Release>:-ffast-math>"
        "$<$<CONFIG:Release>:-march=native>"
    )
endif()

# Global CUDA optimizations
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math")

# --- 5. Include Directories ---
# Internal Project Headers
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# External Headers (Marked as SYSTEM to disable analysis)
include_directories(SYSTEM
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/vendor/imgui
    ${CMAKE_SOURCE_DIR}/src/vendor/imgui/backends
)

# --- 6. Quality Assurance Targets ---
# Clang-Tidy Static Analysis integration
find_program(CLANG_TIDY_EXE NAMES "clang-tidy" "clang-tidy.exe")
if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--extra-arg=/EHsc")
    set(CMAKE_CUDA_CLANG_TIDY "${CLANG_TIDY_EXE};--extra-arg=/EHsc")
else()
    message(WARNING "clang-tidy not found. Code analysis during build is disabled.")
endif()

# Clang-Format Custom Target
find_program(CLANG_FORMAT_EXE 
    NAMES "clang-format" "clang-format.exe"
    PATHS "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/Llvm/x64/bin"
          "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin"
          "D:/Microsoft Visual Studio/2022/Enterprise/VC/Tools/Llvm/x64/bin"
          "D:/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin"
          "C:/Program Files/LLVM/bin"
          "D:/Program Files/LLVM/bin"
)

if(CLANG_FORMAT_EXE)
    message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "src/*.cpp" "src/*.hpp" "src/*.cu" "src/*.cuh"
        "include/*.cpp" "include/*.hpp" "include/*.cu" "include/*.cuh"
        "tests/*.cpp" "tests/*.hpp" "tests/*.cu" "tests/*.cuh"
    )
    # Exclude vendor libraries from formatting
    list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX ".*/vendor/.*")
    
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_SOURCE_FILES}
        COMMENT "Formatting all source/test files (excluding vendor) with clang-format..."
    )
else()
    message(WARNING "clang-format not found. The 'format' target will not be available.")
endif()

# --- 7. Target Definition: Core Logic Library ---
add_library(kmeans_core
    src/backend/cuda_kernels.cu
    src/clustering/engines/base_kmeans_engine.cu
    src/clustering/engines/classical_engine.cu
    src/clustering/engines/quantum_engine.cu
    src/clustering/clustering_manager.cpp
    src/clustering/clustering_factory.cpp
    src/clustering/preprocessor/strided_data_preprocessor.cu
    src/clustering/metrics.cpp
    src/clustering/initializers/random_initializer.cpp
    src/clustering/initializers/kmeans_plus_plus_initializer.cu
    src/io/application.cpp
    src/io/benchmark_runner.cpp
    src/io/benchmark_command.cpp
    src/io/ui_manager.cpp
    src/io/ui/control_panel_ui.cpp
    src/io/ui/video_feed_ui.cpp
    src/io/ui/benchmark_overlay_ui.cpp
)

# Explicitly set CUDA language properties
set_source_files_properties(src/backend/cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/engines/base_kmeans_engine.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/engines/classical_engine.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/engines/quantum_engine.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/initializers/kmeans_plus_plus_initializer.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/preprocessor/strided_data_preprocessor.cu PROPERTIES LANGUAGE CUDA)

set_target_properties(kmeans_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(kmeans_core PUBLIC ${OpenCV_LIBS} glfw)

# Forward complex host flags to the CUDA compiler (NVCC)
if(MSVC)
    target_compile_options(kmeans_core PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Xcompiler=/WX -Xcompiler=/permissive- -Xcompiler=/Zc:preprocessor>
    )
    target_compile_options(kmeans_core PRIVATE
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/fp:fast>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/arch:AVX2>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:W0>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:anglebrackets>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/analyze:external->"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4505>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd6201>"
    )
endif()

# --- 8. Target Definition: Vendor (ImGui) ---
add_library(imgui STATIC
    src/vendor/imgui/imgui.cpp
    src/vendor/imgui/imgui_draw.cpp
    src/vendor/imgui/imgui_tables.cpp
    src/vendor/imgui/imgui_widgets.cpp
    src/vendor/imgui/backends/imgui_impl_glfw.cpp
    src/vendor/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC src/vendor/imgui src/vendor/imgui/backends)
if (MSVC)
    target_compile_options(imgui PRIVATE /analyze-)
endif()
target_link_libraries(imgui PUBLIC glfw)
if (WIN32)
    target_link_libraries(imgui PUBLIC opengl32)
endif()

# Disable code analysis for vendor libraries
set_target_properties(imgui PROPERTIES CXX_CLANG_TIDY "" CUDA_CLANG_TIDY "")

# --- 9. Target Definition: Final Executable ---
target_link_libraries(kmeans_core PUBLIC imgui)
add_executable(kmeans_thesis_segmentation WIN32 src/main.cpp)

if(MSVC)
    # Ensure entry point is correct for a GUI application
    target_link_options(kmeans_thesis_segmentation PRIVATE "LINKER:/ENTRY:mainCRTStartup")
endif()

set_target_properties(kmeans_thesis_segmentation PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(kmeans_thesis_segmentation PRIVATE kmeans_core imgui)
 
# --- 10. Test Orchestration ---
add_subdirectory(tests)
