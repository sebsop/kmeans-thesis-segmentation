#pragma once
#include "clustering/data_preprocessor.hpp"

namespace kmeans {
namespace clustering {

    class FullDataPreprocessor : public DataPreprocessor {
    public:
        FullDataPreprocessor() = default;
        ~FullDataPreprocessor() override = default;

        cv::Mat prepare(const cv::Mat& frame) override;
    };

}
}
