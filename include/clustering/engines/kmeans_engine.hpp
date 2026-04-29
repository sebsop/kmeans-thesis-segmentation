#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"
#include "cuda_runtime.h"

namespace kmeans::clustering {

/**
 * @brief Strategy Design Pattern interface for Execution Engines.
 *
 * Allows seamless transition between classical CPU execution and
 * external quantum simulation environments without affecting the consumer.
 */
class KMeansEngine {
  protected:
    int m_lastIterations = 0;

  public:
    virtual ~KMeansEngine() = default;

    [[nodiscard]] int getLastIterations() const noexcept { return m_lastIterations; }

    /**
     * @brief Executes the clustering algorithm using a CPU-side cv::Mat.
     * Use this when the sample data lives in host memory.
     */
    [[nodiscard]] virtual std::vector<FeatureVector>
    run(const cv::Mat& samples, const std::vector<FeatureVector>& initialCenters, int k,
        int maxIterations) = 0;

    /**
     * @brief GPU-direct path: samples already reside on the device.
     * Skips the H2D upload of samples for maximum throughput.
     * Engines that support this override it; default falls back to run() via D2H.
     */
    [[nodiscard]] virtual std::vector<FeatureVector>
    runOnDevice(float* /*d_samples_ext*/, int /*numPoints*/,
                const std::vector<FeatureVector>& /*initialCenters*/, int /*k*/,
                int /*maxIterations*/) {
        // Default: subclass must override if GPU-direct path is desired
        return {};
    }
};

} // namespace kmeans::clustering
