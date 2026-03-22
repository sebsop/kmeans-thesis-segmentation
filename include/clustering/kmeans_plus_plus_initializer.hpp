#pragma once
#include "clustering/initializer.hpp"
#include <random>

namespace kmeans {
namespace clustering {

    class KMeansPlusPlusInitializer : public Initializer {
    public:
        KMeansPlusPlusInitializer() = default;
        ~KMeansPlusPlusInitializer() override = default;

        std::vector<cv::Vec<float, 5>> initialize(const cv::Mat& samples, int k) const override;
    };

}
}
