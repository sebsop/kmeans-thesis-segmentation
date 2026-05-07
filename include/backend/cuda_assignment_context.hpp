/**
 * @file cuda_assignment_context.hpp
 * @brief Manages GPU memory and execution context for high-performance pixel assignment.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"
#include "cuda_runtime.h"

namespace kmeans::backend {

/**
 * @class CudaAssignmentContext
 * @brief Handles the low-level CUDA resource management for real-time pixel classification.
 *
 * This class implements a zero-copy/pinned-memory pattern to maximize throughput between
 * the CPU (OpenCV) and GPU (CUDA Kernels). It pre-allocates all necessary VRAM and pinned
 * RAM buffers to avoid runtime allocation overhead during the video processing loop.
 */
class CudaAssignmentContext {
  private:
    int m_width;  ///< Width of the frames being processed
    int m_height; ///< Height of the frames being processed
    int m_k;      ///< Number of clusters (centroids)

    cudaStream_t m_stream; ///< Async stream for non-blocking GPU execution

    // Device (GPU) pointers
    unsigned char* m_d_input = nullptr;  ///< Buffer for input frame pixels on GPU
    unsigned char* m_d_output = nullptr; ///< Buffer for segmented labels on GPU
    float* m_d_centers = nullptr;        ///< Buffer for cluster centroids on GPU

    // Pinned (Host) pointers for zero-copy/fast transfer
    unsigned char* m_h_input_pinned = nullptr;  ///< CPU pinned memory for input
    unsigned char* m_h_output_pinned = nullptr; ///< CPU pinned memory for output
    float* m_h_centers_pinned = nullptr;        ///< CPU pinned memory for centroids

    size_t m_imgSize;     ///< Total size in bytes of the input/output images
    size_t m_centersSize; ///< Total size in bytes of the centroids array

  public:
    /**
     * @brief Constructs a context and pre-allocates all necessary GPU/Pinned resources.
     * @param width Image width.
     * @param height Image height.
     * @param k Number of clusters.
     */
    CudaAssignmentContext(int width, int height, int k);

    /**
     * @brief Cleans up all CUDA resources and frees allocated memory.
     */
    ~CudaAssignmentContext() noexcept;

    // Resource management: Disable copying to prevent double-free of GPU handles
    CudaAssignmentContext(const CudaAssignmentContext&) = delete;
    CudaAssignmentContext& operator=(const CudaAssignmentContext&) = delete;

    /** @brief Returns the configured image width. */
    [[nodiscard]] int getWidth() const noexcept { return m_width; }

    /** @brief Returns the number of clusters. */
    [[nodiscard]] int getK() const noexcept { return m_k; }

    /**
     * @brief Executes the pixel-to-centroid assignment pipeline on the GPU.
     *
     * This method:
     * 1. Transfers input data to pinned memory.
     * 2. Launches asynchronous CUDA kernels via the internal stream.
     * 3. Synchronizes and copies the result back to the output Mat.
     *
     * @param frame The input OpenCV Mat (BGR format).
     * @param centers The current cluster centroids.
     * @param output The output OpenCV Mat for the segmented result.
     */
    void run(const cv::Mat& frame, const std::vector<FeatureVector>& centers, cv::Mat& output);
};

} // namespace kmeans::backend