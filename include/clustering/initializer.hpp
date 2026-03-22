#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace kmeans {
namespace clustering {

    class Initializer {
    public:
        virtual ~Initializer() = default;

        virtual std::vector<cv::Vec<float, 5>> initialize(const cv::Mat& samples, int k) const = 0;
    };

} // namespace clustering
} // namespace kmeans
