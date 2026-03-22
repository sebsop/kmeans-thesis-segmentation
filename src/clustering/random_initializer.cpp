#include "clustering/random_initializer.hpp"

namespace kmeans {
namespace clustering {

    std::vector<cv::Vec<float, 5>> RandomInitializer::initialize(const cv::Mat& samples, int k) const {
        CV_Assert(!samples.empty() && k > 0 && k <= samples.rows);
        std::vector<cv::Vec<float, 5>> centers;
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dis(0, samples.rows - 1);

        for (int i = 0; i < k; ++i) {
            int randomRow = dis(gen);
            cv::Vec<float, 5> center;

            for (int d = 0; d < 5; ++d) {
                center[d] = samples.at<float>(randomRow, d);
            }

            centers.push_back(center);
        }
        return centers;
    }

}
}
