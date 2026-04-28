#include "clustering/initializers/random_initializer.hpp"

#include "common/constants.hpp"

namespace kmeans::clustering {

std::vector<cv::Vec<float, constants::FEATURE_DIMS>> RandomInitializer::initialize(const cv::Mat& samples,
                                                                                   int k) const {
    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> centers(k);
    int numPoints = samples.rows;

    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    std::generate(centers.begin(), centers.end(), [&]() {
        int randIdx = dis(gen);
        const auto* rowPtr = samples.ptr<float>(randIdx);
        cv::Vec<float, constants::FEATURE_DIMS> c;
        std::copy_n(rowPtr, constants::FEATURE_DIMS, c.val);
        return c;
    });

    return centers;
}

} // namespace kmeans::clustering
