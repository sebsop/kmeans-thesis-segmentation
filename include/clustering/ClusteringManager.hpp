#pragma once

namespace kmeans {
    class ClusteringManager {
    private:
        std::vector<cv::Vec<float, 5>> m_previousCenters;
        bool m_hasPrevious = false;

    public:
        std::vector<cv::Vec<float, 5>> computeCenters(const cv::Mat& frame, int k)
    };
}