#pragma once
#include "clustering/data_preprocessor.hpp"
#include "core/rcc.hpp"

namespace kmeans {
namespace clustering {

    class RccDataPreprocessor : public DataPreprocessor {
    private:
        RCC m_rcc;

        cv::Mat convertCoresetToMat(const Coreset& coreset);

    public:
        RccDataPreprocessor() = default;
        ~RccDataPreprocessor() override = default;

        cv::Mat prepare(const cv::Mat& frame) override;
    };

}
}
