# =============================================================================
# tests/CMakeLists.txt
# Test suite orchestration for the K-Means Thesis Pipeline.
#
# This file consolidates unit and integration tests into a single master 
# executable, integrates GoogleTest discovery, and provides automation 
# for code coverage analysis.
# =============================================================================

add_executable(all_tests 
    common/vector_math_tests.cpp
    io/ui_integration_tests.cpp
    backend/cuda_kernels_tests.cu
    clustering/engines/base_kmeans_engine_tests.cu
    clustering/engines/classical_engine_tests.cu
    clustering/engines/quantum_engine_tests.cu
    clustering/clustering_factory_tests.cpp
    clustering/clustering_manager_tests.cpp
    clustering/metrics_tests.cpp
    clustering/preprocessor/strided_data_preprocessor_tests.cu
    clustering/initializers/kmeans_plus_plus_initializer_tests.cu
    clustering/initializers/random_initializer_tests.cu
)

# --- 1. Debugging & Analysis Support ---
if(MSVC)
    target_link_options(all_tests PRIVATE /DEBUG:FULL)
    target_compile_options(all_tests PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/Zi /wd6326 /external:W0 /external:anglebrackets /analyze:external->
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi -Xcompiler=/wd6326 -Xcompiler=/external:W0 -Xcompiler=/external:anglebrackets -Xcompiler=/analyze:external->
    )
endif()

set_target_properties(all_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
)

# --- 2. Specialized Static Analysis for Tests ---
# Disable the "convert to static" check because GTest methods are designed to be virtual overrides.
if(CLANG_TIDY_EXE)
    set_target_properties(all_tests PROPERTIES 
        CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=-readability-convert-member-functions-to-static;--extra-arg=/EHsc"
    )
endif()

target_link_libraries(all_tests PRIVATE kmeans_core GTest::gtest_main)

# --- 3. GTest Discovery Integration ---
# Automatically find all TEST() and TEST_F() macros and register them with CTest.
include(GoogleTest)
gtest_discover_tests(all_tests)
