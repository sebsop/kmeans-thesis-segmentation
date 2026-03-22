#include "clustering/full_data_preprocessor.hpp"
#include "common/constants.hpp"

namespace kmeans {
namespace clustering {

    cv::Mat FullDataPreprocessor::prepare(const cv::Mat& frame) {
        CV_Assert(!frame.empty());
        int rows = frame.rows;
        int cols = frame.cols;
        int totalPixels = rows * cols;

        cv::Mat samples(totalPixels, 5, CV_32F);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int idx = r * cols + c;
                cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);

                float x01 = static_cast<float>(c) / cols;
                float y01 = static_cast<float>(r) / rows;

                samples.at<float>(idx, 0) = pixel[0] * kmeans::COLOR_SCALE;
                samples.at<float>(idx, 1) = pixel[1] * kmeans::COLOR_SCALE;
                samples.at<float>(idx, 2) = pixel[2] * kmeans::COLOR_SCALE;
                samples.at<float>(idx, 3) = x01 * kmeans::SPATIAL_SCALE;
                samples.at<float>(idx, 4) = y01 * kmeans::SPATIAL_SCALE;
            }
        }
        return samples;
    }

}
}
