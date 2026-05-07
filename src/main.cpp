/**
 * @file main.cpp
 * @brief Application entry point.
 */

#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/core/utils/logger.hpp>

#include "common/config.hpp"
#include "io/application.hpp"

using namespace kmeans;

/**
 * @brief Main entry point for the K-Means Segmentation Benchmark.
 *
 * Initializes the high-level application context and handles any unhandled
 * exceptions at the top level to ensure the console output remains informative.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int EXIT_SUCCESS on clean exit, EXIT_FAILURE on fatal error.
 */
int main(int argc, char** argv) {
    try {
        // Suppress verbose OpenCV logs to maintain a professional console output
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        io::Application app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal UI Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Fatal Unknown Error!\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
