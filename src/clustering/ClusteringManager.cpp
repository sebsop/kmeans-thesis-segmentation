#include "clustering/ClusteringManager.hpp"

namespace kmeans {
    std::vector<cv::Vec<float, 5>> ClusteringManager::computeCenters(const cv::Mat& frame, int k) {
        std::vector<cv::Vec<float, 5>> initialCenters;

        if (m_hasPrevious && m_previousCenters.size() == k) {
            // Use centers from the last frame as the starting point
            initialCenters = m_previousCenters;
        }
        else {
            // Fallback to K-Means++ for the first frame or if k changes
            initialCenters = Initialization::pickKMeansPlusPlus(frame, k);
        }

        // Run the K-Means loop (Classical or Quantum)
        m_previousCenters = runKMeansLoop(frame, initialCenters);
        m_hasPrevious = true;

        return m_previousCenters;
    }
}