#pragma once

#include <opencv2/imgproc.hpp>

#include "clustering/preprocessors/data_preprocessor.hpp"

namespace kmeans::clustering {

class FullDataPreprocessor final : public DataPreprocessor {
  public:
    FullDataPreprocessor() = default;
    ~FullDataPreprocessor();

    /** @brief CPU path: runs GPU kernel and downloads result to a cv::Mat (for RCC compat). */
    [[nodiscard]] cv::Mat prepare(const cv::Mat& frame) override final;

    /**
     * @brief GPU-direct path: runs the GPU kernel but does NOT download results.
     * Returns the raw device pointer to the sample features already on GPU.
     * The pointer is valid until the next call to prepareDevice() or prepare().
     */
    [[nodiscard]] float* prepareDevice(const cv::Mat& frame, int& outNumPoints);

    /** @brief Downloads the current GPU-side samples to a cv::Mat. */
    [[nodiscard]] cv::Mat download() const;

    [[nodiscard]] int getNumSamples() const noexcept { return m_cached_n; }

    void reset() override final;

  private:
    int m_cached_n = 0;
    void* m_d_frame_data = nullptr;
    void* m_d_samples = nullptr;

    void uploadAndRun(const cv::Mat& frame);
};

} // namespace kmeans::clustering
