/**
 * @file strided_data_preprocessor.cu
 * @brief High-performance GPU-accelerated image-to-feature conversion.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

#include "clustering/preprocessor/strided_data_preprocessor.hpp"
#include "common/constants.hpp"
#include "common/utils.hpp"

/** @brief Macro for internal CUDA error handling in the preprocessor. */
#define CUDA_CHECK_PREP(call)                                                                                          \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

/**
 * @brief CUDA kernel to extract normalized 5D features from raw image pixels.
 *
 * This kernel iterates over the output grid and samples the corresponding
 * pixels from the input image using a 'stride'. It converts BGR values
 * and spatial coordinates into a normalized [0, 1] range.
 *
 * @param frame_data Input raw BGR pixel data.
 * @param samples [Output] Matrix of 5D feature vectors.
 * @param cols Input image width.
 * @param rows Input image height.
 * @param stride The sampling interval.
 * @param out_cols Number of sampled columns.
 * @param out_rows Number of sampled rows.
 * @param invCols Reciprocal of width (for normalization).
 * @param invRows Reciprocal of height (for normalization).
 * @param color_scale Scaling factor for BGR channels.
 * @param spatial_scale Scaling factor for XY coordinates.
 */
__global__ void preprocess_strided_kernel(const uchar3* __restrict__ frame_data, float* __restrict__ samples, int cols,
                                          int rows, int stride, int out_cols, int out_rows, float invCols,
                                          float invRows, float color_scale, float spatial_scale) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_cols && out_y < out_rows) {
        int x = out_x * stride;
        int y = out_y * stride;

        int in_idx = y * cols + x;
        int out_idx = out_y * out_cols + out_x;

        uchar3 bgr = frame_data[in_idx];

        float x01 = static_cast<float>(x) * invCols;
        float y01 = static_cast<float>(y) * invRows;

        // Store as [B, G, R, X, Y] feature vector
        samples[out_idx * constants::clustering::FEATURE_DIMS + 0] = static_cast<float>(bgr.x) * color_scale;
        samples[out_idx * constants::clustering::FEATURE_DIMS + 1] = static_cast<float>(bgr.y) * color_scale;
        samples[out_idx * constants::clustering::FEATURE_DIMS + 2] = static_cast<float>(bgr.z) * color_scale;
        samples[out_idx * constants::clustering::FEATURE_DIMS + 3] = x01 * spatial_scale;
        samples[out_idx * constants::clustering::FEATURE_DIMS + 4] = y01 * spatial_scale;
    }
}

StridedDataPreprocessor::~StridedDataPreprocessor() {
    try {
        reset();
    } catch (...) {
        // Destructors must not throw.
    }
}

/**
 * @brief Releases allocated GPU buffers.
 */
void StridedDataPreprocessor::reset() {
    if (m_d_frame_data) {
        CUDA_CHECK_PREP(cudaFree(m_d_frame_data));
        m_d_frame_data = nullptr;
    }
    if (m_d_samples) {
        CUDA_CHECK_PREP(cudaFree(m_d_samples));
        m_d_samples = nullptr;
    }
    m_cached_n = 0;
    m_extracted_points = 0;
}

/**
 * @brief Orchestrates the memory upload and kernel execution.
 *
 * @param frame The input image from the camera.
 * @param stride The sampling stride to apply.
 */
void StridedDataPreprocessor::uploadAndRun(const cv::Mat& frame, int stride) {
    CV_Assert(frame.type() == CV_8UC3 && frame.isContinuous());
    CV_Assert(stride >= 1);

    int n = frame.rows * frame.cols;

    float invCols = 1.0f / static_cast<float>(frame.cols);
    float invRows = 1.0f / static_cast<float>(frame.rows);

    int out_cols = (frame.cols + stride - 1) / stride;
    int out_rows = (frame.rows + stride - 1) / stride;

    // Buffer management: only reallocate if resolution changes
    if (n != m_cached_n) {
        reset();
        CUDA_CHECK_PREP(cudaMalloc(&m_d_frame_data, n * sizeof(uchar3)));
        CUDA_CHECK_PREP(cudaMalloc(&m_d_samples, n * constants::clustering::FEATURE_DIMS * sizeof(float)));
        m_cached_n = n;
    }

    m_extracted_points = out_cols * out_rows;

    CUDA_CHECK_PREP(cudaMemcpy(m_d_frame_data, frame.ptr<uchar3>(), n * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockSize(constants::cuda::BLOCK_2D_X, constants::cuda::BLOCK_2D_Y);
    dim3 gridSize(common::calculateGridDim(out_cols, blockSize.x), common::calculateGridDim(out_rows, blockSize.y));

    preprocess_strided_kernel<<<gridSize, blockSize>>>(
        static_cast<uchar3*>(m_d_frame_data), static_cast<float*>(m_d_samples), frame.cols, frame.rows, stride,
        out_cols, out_rows, invCols, invRows, kmeans::constants::video::COLOR_SCALE,
        kmeans::constants::video::SPATIAL_WEIGHT);

    CUDA_CHECK_PREP(cudaDeviceSynchronize());
}

cv::Mat StridedDataPreprocessor::prepare(const cv::Mat& frame) {
    uploadAndRun(frame, 1);
    return download();
}

/**
 * @brief Prepares data on the GPU and returns a device pointer.
 *
 * This is the preferred method for the high-performance pipeline, as it
 * avoids downloading the features back to the CPU.
 */
float* StridedDataPreprocessor::prepareDevice(const cv::Mat& frame, int stride, int& outNumPoints) {
    uploadAndRun(frame, stride);
    outNumPoints = m_extracted_points;
    return static_cast<float*>(m_d_samples);
}

cv::Mat StridedDataPreprocessor::download() const {
    cv::Mat samples(m_extracted_points, constants::clustering::FEATURE_DIMS, CV_32F);
    if (m_extracted_points > 0 && m_d_samples) {
        CUDA_CHECK_PREP(cudaMemcpy(samples.ptr<float>(), m_d_samples,
                                   m_extracted_points * constants::clustering::FEATURE_DIMS * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    }
    return samples;
}

} // namespace kmeans::clustering
