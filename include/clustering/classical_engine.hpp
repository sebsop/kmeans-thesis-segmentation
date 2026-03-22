#pragma once
#include "clustering/kmeans_engine.hpp"
#include "common/constants.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace kmeans {

    class ClassicalEngine : public KMeansEngine {
    public:
        ClassicalEngine() = default;
        virtual ~ClassicalEngine() = default;

        std::vector<cv::Vec<float, 5>> run(
            const cv::Mat& samples,
            const std::vector<cv::Vec<float, 5>>& initialCenters,
            int k) override;
    };

} // namespace kmeans