#pragma once
#include <opencv2/core.hpp>

namespace kmeans {
namespace clustering {

    class DataPreprocessor {
    public:
        virtual ~DataPreprocessor() = default;

        virtual cv::Mat prepare(const cv::Mat& frame) = 0;
    };

}
}
