#pragma once
#include "clustering/data_preprocessor.hpp"

namespace kmeans {
namespace clustering {

    class FlatDataPreprocessor : public DataPreprocessor {
    public:
        FlatDataPreprocessor() = default;
        ~FlatDataPreprocessor() override = default;

        cv::Mat prepare(const cv::Mat& frame) override;
    };

}
}
