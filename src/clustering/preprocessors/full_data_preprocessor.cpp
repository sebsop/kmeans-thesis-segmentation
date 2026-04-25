#include "clustering/preprocessors/full_data_preprocessor.hpp"

#include "common/utils.hpp"

namespace kmeans::clustering {

cv::Mat FullDataPreprocessor::prepare(const cv::Mat& frame) {
    int n = frame.rows * frame.cols;
    cv::Mat samples(n, 5, CV_32F);

    float invCols = 1.0f / static_cast<float>(frame.cols);
    float invRows = 1.0f / static_cast<float>(frame.rows);

    float* outPtr = samples.ptr<float>(0);

    for (int r = 0; r < frame.rows; ++r) {
        const auto* rowPtr = frame.ptr<cv::Vec3b>(r);
        float y01 = static_cast<float>(r) * invRows;
        
        for (int c = 0; c < frame.cols; ++c) {
            const cv::Vec3b& bgr = rowPtr[c];
            float x01 = static_cast<float>(c) * invCols;

            cv::Vec<float, 5> feature = common::makeFeature(cv::Vec3f(bgr[0], bgr[1], bgr[2]), x01, y01);

            *outPtr++ = feature[0];
            *outPtr++ = feature[1];
            *outPtr++ = feature[2];
            *outPtr++ = feature[3];
            *outPtr++ = feature[4];
        }
    }

    return samples;
}

} // namespace kmeans::clustering
