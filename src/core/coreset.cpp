#include "core/coreset.hpp"

#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

#include "common/constants.hpp"

namespace kmeans::core {

Coreset buildCoresetFromFrame(const cv::Mat& frame) {
    Coreset coreset;

    int rows = frame.rows;
    int cols = frame.cols;
    int total_pixels = rows * cols;

    std::random_device random_device_instance;
    std::mt19937 gen(random_device_instance());
    std::uniform_int_distribution<> row_dist(0, rows - 1);
    std::uniform_int_distribution<> col_dist(0, cols - 1);

    coreset.points.reserve(constants::SAMPLE_COUNT);

    for (int i = 0; i < constants::SAMPLE_COUNT; ++i) {
        int sampled_row = row_dist(gen);
        int sampled_col = col_dist(gen);
        cv::Vec3b pixel = frame.at<cv::Vec3b>(sampled_row, sampled_col);

        CoresetPoint point;
        point.bgr = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
        point.weight = static_cast<float>(total_pixels) / static_cast<float>(constants::SAMPLE_COUNT);
        point.x = static_cast<float>(sampled_col) / static_cast<float>(cols);
        point.y = static_cast<float>(sampled_row) / static_cast<float>(rows);

        coreset.points.push_back(point);
    }

    return coreset;
}

Coreset mergeCoresets(const Coreset& CoresetA, const Coreset& CoresetB) {
    Coreset merged;

    merged.points.reserve(CoresetA.points.size() + CoresetB.points.size());
    merged.points.insert(merged.points.end(), CoresetA.points.begin(), CoresetA.points.end());
    merged.points.insert(merged.points.end(), CoresetB.points.begin(), CoresetB.points.end());

    // If the merged coreset exceeds the sample size, randomly downsample it
    if (merged.points.size() > constants::SAMPLE_COUNT) {
        std::shuffle(merged.points.begin(), merged.points.end(), std::mt19937{std::random_device{}()});
        merged.points.resize(constants::SAMPLE_COUNT);
    }

    return merged;
}

} // namespace kmeans::core