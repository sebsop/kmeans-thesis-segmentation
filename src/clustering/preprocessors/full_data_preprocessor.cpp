#include "clustering/preprocessors/full_data_preprocessor.hpp"

#include "common/utils.hpp"

namespace kmeans::clustering {

cv::Mat FullDataPreprocessor::prepare(const cv::Mat& frame) {
    int n = frame.rows * frame.cols;
    cv::Mat samples(n, 5, CV_32F);

    int idx = 0;
    for (int r = 0; r < frame.rows; ++r) {
        const auto* rowPtr = frame.ptr<cv::Vec3b>(r);
        for (int c = 0; c < frame.cols; ++c) {
            const cv::Vec3b& bgr = rowPtr[c];
            float x01 = static_cast<float>(c) / static_cast<float>(frame.cols);
            float y01 = static_cast<float>(r) / static_cast<float>(frame.rows);

            cv::Vec<float, 5> feature = common::makeFeature(cv::Vec3f(bgr[0], bgr[1], bgr[2]), x01, y01);

            for (int d = 0; d < 5; ++d) {
                samples.at<float>(idx, d) = feature[d];
            }
            idx++;
        }
    }
    return samples;
}

} // namespace kmeans::clustering
