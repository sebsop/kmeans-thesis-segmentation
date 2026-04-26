#include <iostream>

#include <opencv2/core/utils/logger.hpp>

#include "common/config.hpp"
#include "io/application.hpp"

using namespace kmeans;

int main(int argc, char** argv) {
    try {
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
