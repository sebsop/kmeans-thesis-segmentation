#include "clustering/initializers/random_initializer.hpp"

namespace kmeans::clustering {

std::vector<cv::Vec<float, 5>> RandomInitializer::initialize(const cv::Mat& samples, int k) const {
    std::vector<cv::Vec<float, 5>> centers(k);
    int numPoints = samples.rows;

    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    for (int i = 0; i < k; ++i) {
        int randIdx = dis(gen);
        const auto* rowPtr = samples.ptr<float>(randIdx);
        centers[i] = cv::Vec<float, 5>(rowPtr[0], rowPtr[1], rowPtr[2], rowPtr[3], rowPtr[4]);
    }

    return centers;
}

} // namespace kmeans::clustering
