#include "clustering.hpp"
#include "clustering_backends.hpp"
#include "enums.hpp"

namespace kmeans {
    // Entry point to segment a frame using K-means clustering with different algorithms
    cv::Mat segmentFrameWithKMeans(
        Algorithm algorithm,
        const cv::Mat& frame,
        int k,
        int sample_size,
        float color_scale,
        float spatial_scale)
    {
        // Dispatch to the appropriate algorithm
        switch (algorithm) {
        case Algorithm::KMEANS:
            return segmentFrameWithKMeans_regular(frame, k, sample_size, Initialization::RANDOM, color_scale, spatial_scale);
        case Algorithm::KMEANS_PLUSPLUS:
            return segmentFrameWithKMeans_regular(frame, k, sample_size, Initialization::KMEANS_PLUSPLUS, color_scale, spatial_scale);
        case Algorithm::KMEANS_QUANTUM:
            return segmentFrameWithKMeans_quantum(frame, k, sample_size, color_scale, spatial_scale);
        default:
            throw std::invalid_argument("Unknown algorithm type in segmentFrameWithKMeans");
        }
    }
}
