#pragma once
#include <opencv2/core.hpp>

namespace kmeans {
    class KMeansEngine {
    public:
        virtual ~KMeansEngine() = default;

        virtual std::vector<cv::Vec<float, 5>> run(
            const cv::Mat& samples,
            const std::vector<cv::Vec<float, 5>>& initialCenters,
            int k) = 0;
    };
}