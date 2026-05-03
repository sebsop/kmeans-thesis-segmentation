#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/core/utils/logger.hpp>

#include "common/config.hpp"
#include "io/application.hpp"

using namespace kmeans;

int main(int argc, char** argv) {
    try {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            std::cerr << "Error: No CUDA-capable GPU detected or CUDA driver is not installed properly.\n";
            std::cerr << "This application requires a CUDA-capable GPU to run.\n";
            return EXIT_FAILURE;
        }

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
