#include "clustering/kmeans_plus_plus_initializer.hpp"
#include <algorithm>

namespace kmeans {
namespace clustering {

    std::vector<cv::Vec<float, 5>> KMeansPlusPlusInitializer::initialize(const cv::Mat& samples, int k) const {
        CV_Assert(!samples.empty() && k > 0 && k <= samples.rows);
        std::vector<cv::Vec<float, 5>> centers;
        centers.reserve(k);

        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dis(0, samples.rows - 1);

        // 1. Pick the first center completely at random
        centers.push_back(samples.at<cv::Vec<float, 5>>(dis(gen)));

        // 2. Pick the remaining k-1 centers
        for (int i = 1; i < k; ++i) {
            std::vector<float> distancesSq(samples.rows);
            float sumDistSq = 0.0f;

            // For each point, find the distance to the NEAREST existing center
            for (int p = 0; p < samples.rows; ++p) {
                float minD2 = 1e20f;
                cv::Vec<float, 5> point = samples.at<cv::Vec<float, 5>>(p);

                for (const auto& c : centers) {
                    float d2 = cv::norm(point - c, cv::NORM_L2SQR);
                    if (d2 < minD2) minD2 = d2;
                }
                distancesSq[p] = minD2;
                sumDistSq += minD2;
            }

            // Weighted probability distribution: pick next center proportional to distance squared
            std::uniform_real_distribution<float> probDis(0, sumDistSq);
            float threshold = probDis(gen);
            float currentSum = 0.0f;

            for (int p = 0; p < samples.rows; ++p) {
                currentSum += distancesSq[p];
                if (currentSum >= threshold) {
                    centers.push_back(samples.at<cv::Vec<float, 5>>(p));
                    break;
                }
            }
        }
        return centers;
    }

}
}
