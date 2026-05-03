# Master Test Suite (All tests in one executable)
add_executable(all_tests 
    backend/cuda_kernels_tests.cu
    clustering/engines/base_engine_tests.cu
    clustering/engines/classical_engine_tests.cu
    clustering/engines/quantum_engine_tests.cu
)

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
