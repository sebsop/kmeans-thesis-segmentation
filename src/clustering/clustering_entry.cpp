#include "clustering.hpp"
#include "clustering_backends.hpp"
#include "enums.hpp"

namespace kmeans {
    // Entry point to segment a frame using K-means clustering with different algorithms
    cv::Mat segmentFrameWithKMeans(
        Algorithm algorithm,
        const cv::Mat& frame,
        int k)
    {
        // Dispatch to the appropriate algorithm
        switch (algorithm) {
        case Algorithm::KMEANS_REGULAR:
            return segmentFrameWithKMeans_regular(frame, k, Initialization::RANDOM);
        case Algorithm::KMEANS_QUANTUM:
            return segmentFrameWithKMeans_quantum(frame, k);
        default:
            throw std::invalid_argument("Unknown algorithm type in segmentFrameWithKMeans");
        }
    }
}
