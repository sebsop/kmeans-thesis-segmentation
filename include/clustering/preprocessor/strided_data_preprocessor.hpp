#pragma once

#include <memory>

#include <opencv2/core/mat.hpp>

/**
 * @file strided_data_preprocessor.hpp
 * @brief GPU-accelerated image-to-feature transformation.
 */

namespace kmeans::clustering {

/**
 * @class StridedDataPreprocessor
 * @brief Transforms raw video frames into a normalized 5D feature space.
 *
 * This class is responsible for the critical first step of the pipeline:
 * Converting an OpenCV BGR image into a format suitable for clustering.
 * For each selected pixel, it extracts:
 * - Color: Red, Green, Blue (normalized to 0-1)
 * - Position: X, Y (normalized by image dimensions to 0-1)
 *
 * It uses a 'stride' parameter to sample every N-th pixel, significantly
 * reducing the computational load for high-resolution video while
 * maintaining the overall visual structure.
 */
class StridedDataPreprocessor {
  public:
    StridedDataPreprocessor() = default;
    ~StridedDataPreprocessor();

    /**
     * @brief High-level preparation. Returns feature data to CPU.
     * @param frame Input BGR frame.
     * @return cv::Mat where each row is a 5D feature vector.
     */
    cv::Mat prepare(const cv::Mat& frame);

    /** @brief Frees allocated GPU resources and resets internal state. */
    void reset();

    /**
     * @brief GPU-Direct preparation.
     *
     * Performs the transformation on the GPU and keeps the result in VRAM.
     * This is the preferred path for the real-time pipeline to avoid
     * unnecessary Device-to-Host transfers.
     *
     * @param frame Input BGR frame.
     * @param stride Sampling rate (1 = every pixel, 2 = every 2nd, etc).
     * @param outNumPoints [out] Returns the number of features extracted.
     * @return Pointer to the 5D feature buffer in GPU memory.
     */
    float* prepareDevice(const cv::Mat& frame, int stride, int& outNumPoints);

    /**
     * @brief Downloads the current GPU feature buffer to CPU memory.
     * @return cv::Mat of the extracted features.
     */
    cv::Mat download() const;

  private:
    /** @brief Internal helper to launch the CUDA preprocessing kernels. */
    void uploadAndRun(const cv::Mat& frame, int stride);

    void* m_d_frame_data = nullptr; ///< Raw frame data in VRAM (uchar3)
    float* m_d_samples = nullptr;   ///< Transformed 5D feature buffer in VRAM
    int m_cached_n = 0;             ///< Number of pixels in the cached frame (for resize detection)
    int m_extracted_points = 0;     ///< Actual number of features after striding
};

} // namespace kmeans::clustering
