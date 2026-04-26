#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "cuda_runtime.h"

namespace kmeans::clustering {

/**
 * @brief Strategy Design Pattern interface for Execution Engines.
 *
 * Allows seamless transition between classical CPU execution and
 * external quantum simulation environments without affecting the consumer.
 */
class KMeansEngine {
  public:
    virtual ~KMeansEngine() = default;

    /** @brief Executes the clustering algorithm using the specific engine implementation. */
    [[nodiscard]] virtual std::vector<cv::Vec<float, 5>>
    run(const cv::Mat& samples, const std::vector<cv::Vec<float, 5>>& initialCenters, int k) = 0;

    /**
     * @brief GPU-direct path: samples already reside on the device.
     * Skips the H2D upload of samples for maximum throughput.
     * Engines that support this override it; default falls back to run() via D2H.
     */
    [[nodiscard]] virtual std::vector<cv::Vec<float, 5>>
    runOnDevice(float* /*d_samples_ext*/, int /*numPoints*/,
                const std::vector<cv::Vec<float, 5>>& /*initialCenters*/, int /*k*/) {
        // Default: subclass must override if GPU-direct path is desired
        return {};
    }
};

} // namespace kmeans::clustering
