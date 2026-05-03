add_executable(all_tests 
    common/utils_tests.cpp
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

if(MSVC)
    target_link_options(all_tests PRIVATE /PROFILE /DEBUG:FULL)
    target_compile_options(all_tests PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/Zi>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
    )
endif()

set_target_properties(all_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(all_tests PRIVATE kmeans_core GTest::gtest_main)

# Register the master test with GTest discovery
include(GoogleTest)
gtest_discover_tests(all_tests)

# Automation for Code Coverage (Windows / OpenCppCoverage)
find_program(OPENCPP_COVERAGE OpenCppCoverage)
if(OPENCPP_COVERAGE)
    add_custom_target(generate_coverage
        COMMAND ${OPENCPP_COVERAGE} 
            --sources "${CMAKE_SOURCE_DIR}/src" 
            --sources "${CMAKE_SOURCE_DIR}/include"
            --excluded_sources "${CMAKE_SOURCE_DIR}/src/vendor"
            --export_html "${CMAKE_BINARY_DIR}/CoverageReport"
            --aggregate_by_file
            -- ./all_tests.exe
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating Master Code Coverage Report..."
    )
endif()
