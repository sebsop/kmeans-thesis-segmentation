#include "coreset.hpp"
#include "constants.hpp"
#include <opencv2/opencv.hpp>
#include <random>
#include <iostream>

namespace kmeans {
    // Build a coreset by randomly sampling `sample_size` pixels from the frame and 
    Coreset buildCoresetFromFrame(const cv::Mat& frame)
    {
        Coreset coreset;

        int rows = frame.rows;
        int cols = frame.cols;
        int total_pixels = rows * cols;

        std::random_device rd; // Declare a random number generator
        std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd() (period is a Mersenne prime (prime of the form 2^n - 1))
        std::uniform_int_distribution<> row_dist(0, rows - 1); // Get an uniformly random int
        std::uniform_int_distribution<> col_dist(0, cols - 1);

        for (int i = 0; i < SAMPLE_COUNT; ++i) {
            int r = row_dist(gen); // Get the index of a random pixel
            int c = col_dist(gen);
            cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c); // Sample the pixel value. Returns a Vec3b (3-channel uchar (0-255) vector)

            CoresetPoint pt;
            pt.bgr = cv::Vec3f(pixel[0], pixel[1], pixel[2]); // By default, OpenCV stores images in the BGR order
            pt.weight = float(total_pixels) / float(SAMPLE_COUNT); // Each point has a weight associated to it, represented by the amount
            pt.x = float(c) / float(cols);                        // of original pixels it represents
            pt.y = float(r) / float(rows); // Normalize coordinates to [0, 1] range

            coreset.points.push_back(pt);
        }

        return coreset;
    }

    Coreset mergeCoresets(const Coreset& A, const Coreset& B)
    {
        Coreset merged;

        merged.points.insert(merged.points.end(), A.points.begin(), A.points.end());
        merged.points.insert(merged.points.end(), B.points.begin(), B.points.end());

        // If the merged coreset exceeds the sample size, randomly downsample it
        if (merged.points.size() > SAMPLE_COUNT) {
            std::shuffle(merged.points.begin(), merged.points.end(), std::mt19937{ std::random_device{}() });
            merged.points.resize(SAMPLE_COUNT);
        }

        return merged;
    }
}