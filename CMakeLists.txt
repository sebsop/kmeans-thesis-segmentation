cmake_minimum_required(VERSION 3.18)
project(KMeansThesisSegmentation LANGUAGES CXX CUDA)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(glfw3 REQUIRED)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure Release build by default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
endif()

# Enable LTO/IPO
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Compiler optimizations (Only applied in Release mode)
if(MSVC)
    add_compile_options(
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/fp:fast>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/arch:AVX2>"
        # Suppress warnings and code analysis in external headers (C++ only)
        "$<$<COMPILE_LANGUAGE:CXX>:/external:W0>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>"
        "$<$<COMPILE_LANGUAGE:CXX>:/analyze:external->"
        # Ensure exceptions are enabled for clang-tidy (C++ only)
        "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
    )
else()
    add_compile_options(
        "$<$<CONFIG:Release>:-O3>"
        "$<$<CONFIG:Release>:-ffast-math>"
        "$<$<CONFIG:Release>:-march=native>"
    )
endif()

# CUDA Optimizations (Only applied in Release mode)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math")

# Project include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# External (SYSTEM) include directories to suppress compiler/analysis warnings
include_directories(SYSTEM
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/vendor/imgui
    ${CMAKE_SOURCE_DIR}/src/vendor/imgui/backends
)

# Enable Clang-Tidy during build if available
find_program(CLANG_TIDY_EXE NAMES "clang-tidy" "clang-tidy.exe")
if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--extra-arg=/EHsc")
    set(CMAKE_CUDA_CLANG_TIDY "${CLANG_TIDY_EXE};--extra-arg=/EHsc")
else()
    message(WARNING "clang-tidy not found. Code analysis during build is disabled.")
endif()

# Custom target to format all source files (excluding vendor libraries)
find_program(CLANG_FORMAT_EXE NAMES "clang-format" "clang-format.exe")
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "src/*.cpp" "src/*.hpp" "src/*.cu" "src/*.cuh"
        "include/*.cpp" "include/*.hpp" "include/*.cu" "include/*.cuh"
    )
    # Remove all files inside src/vendor/ from the list
    list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX ".*/vendor/.*")
    
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_SOURCE_FILES}
        COMMENT "Formatting all source files (excluding vendor) with clang-format..."
    )
endif()

# Core Logic Library
add_library(kmeans_core
    src/backend/cuda_kernels.cu
    src/clustering/engines/classical_engine.cu
    src/clustering/engines/quantum_engine.cu
    src/clustering/clustering_manager.cpp
    src/clustering/clustering_factory.cpp
    src/clustering/preprocessor/strided_data_preprocessor.cu
    src/clustering/metrics.cpp
    src/clustering/initializers/random_initializer.cpp
    src/clustering/initializers/kmeans_plus_plus_initializer.cu
    src/common/utils.cpp
    src/io/application.cpp
)

set_source_files_properties(src/backend/cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/engines/classical_engine.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/engines/quantum_engine.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/initializers/kmeans_plus_plus_initializer.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/clustering/preprocessor/full_data_preprocessor.cu PROPERTIES LANGUAGE CUDA)

set_target_properties(kmeans_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON       # Enable separable compilation for CUDA
    CUDA_STANDARD 17                    # Use C++17 standard for CUDA files
    CUDA_STANDARD_REQUIRED ON           # Enforce the use of the specified C++ standard
    CUDA_ARCHITECTURES "native"         # Specify target architecture (native for the current machine)
)

target_link_libraries(kmeans_core PUBLIC ${OpenCV_LIBS} glfw cppzmq)

# Forward a minimal set of host MSVC flags into nvcc for the CUDA sources of this target.
# Each host flag is passed as its own `-Xcompiler` entry to avoid nvcc command-line parsing issues
# that arise when multiple flags are combined/quoted into a single `-Xcompiler` argument.
if(MSVC)
    target_compile_options(kmeans_core PRIVATE
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/fp:fast>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/arch:AVX2>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:W0>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:anglebrackets>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/analyze:external->"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>"
    )
endif()

# ImGui Library
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

# Disable Clang-Tidy for third-party libraries
set_target_properties(imgui PROPERTIES CXX_CLANG_TIDY "" CUDA_CLANG_TIDY "")

# Bind ImGui exactly into the core logic library so application.cpp can inherit the headers
target_link_libraries(kmeans_core PUBLIC imgui)

add_executable(kmeans_thesis_segmentation WIN32 src/main.cpp)

if(MSVC)
    target_link_options(kmeans_thesis_segmentation PRIVATE "/ENTRY:mainCRTStartup")
endif()

# Set CUDA properties for the main executable
set_target_properties(kmeans_thesis_segmentation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(kmeans_thesis_segmentation PRIVATE kmeans_core imgui)
