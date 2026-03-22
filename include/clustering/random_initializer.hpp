#pragma once
#include "clustering/initializer.hpp"
#include <random>

namespace kmeans {
namespace clustering {

    class RandomInitializer : public Initializer {
    public:
        RandomInitializer() = default;
        ~RandomInitializer() override = default;

        std::vector<cv::Vec<float, 5>> initialize(const cv::Mat& samples, int k) const override;
    };

}
}
