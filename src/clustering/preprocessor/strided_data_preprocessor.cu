#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

#include "common/constants.hpp"
#include "clustering/preprocessor/strided_data_preprocessor.hpp"

#define CUDA_CHECK_PREP(call)                                                                                          \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

__global__ void preprocess_strided_kernel(const uchar3* __restrict__ frame_data, float* __restrict__ samples, 
                                            int cols, int rows, int stride, 
                                            int out_cols, int out_rows,
                                            float invCols, float invRows, 
                                            float color_scale, float spatial_scale) {
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

        samples[out_idx * 5 + 0] = static_cast<float>(bgr.x) * color_scale;
        samples[out_idx * 5 + 1] = static_cast<float>(bgr.y) * color_scale;
        samples[out_idx * 5 + 2] = static_cast<float>(bgr.z) * color_scale;
        samples[out_idx * 5 + 3] = x01 * spatial_scale;
        samples[out_idx * 5 + 4] = y01 * spatial_scale;
    }
}

StridedDataPreprocessor::~StridedDataPreprocessor() {
    reset();
}

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

void StridedDataPreprocessor::uploadAndRun(const cv::Mat& frame, int stride) {
    CV_Assert(frame.type() == CV_8UC3 && frame.isContinuous());
    CV_Assert(stride >= 1);

    int n = frame.rows * frame.cols;

    float invCols = 1.0f / static_cast<float>(frame.cols);
    float invRows = 1.0f / static_cast<float>(frame.rows);

    int out_cols = (frame.cols + stride - 1) / stride;
    int out_rows = (frame.rows + stride - 1) / stride;
    // Allocate memory if frame size changes (e.g. initial setup)
    if (n != m_cached_n) {
        reset();
        CUDA_CHECK_PREP(cudaMalloc(&m_d_frame_data, n * sizeof(uchar3)));
        CUDA_CHECK_PREP(cudaMalloc(&m_d_samples, n * 5 * sizeof(float))); // Allocate full N to be safe against stride=1
        m_cached_n = n;
    }

    m_extracted_points = out_cols * out_rows;

    CUDA_CHECK_PREP(cudaMemcpy(m_d_frame_data, frame.ptr<uchar3>(), n * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((out_cols + blockSize.x - 1) / blockSize.x, (out_rows + blockSize.y - 1) / blockSize.y);

    preprocess_strided_kernel<<<gridSize, blockSize>>>(
        static_cast<uchar3*>(m_d_frame_data), static_cast<float*>(m_d_samples), 
        frame.cols, frame.rows, stride, out_cols, out_rows,
        invCols, invRows, kmeans::constants::COLOR_SCALE, kmeans::constants::SPATIAL_SCALE);

    CUDA_CHECK_PREP(cudaDeviceSynchronize());
}

cv::Mat StridedDataPreprocessor::prepare(const cv::Mat& frame) {
    uploadAndRun(frame, 1); // Fallback standard CPU prepare uses stride 1
    return download();
}

float* StridedDataPreprocessor::prepareDevice(const cv::Mat& frame, int stride, int& outNumPoints) {
    uploadAndRun(frame, stride);
    outNumPoints = m_extracted_points;
    return static_cast<float*>(m_d_samples);
}

cv::Mat StridedDataPreprocessor::download() const {
    cv::Mat samples(m_extracted_points, 5, CV_32F);
    if (m_extracted_points > 0 && m_d_samples) {
        CUDA_CHECK_PREP(cudaMemcpy(samples.ptr<float>(), m_d_samples, m_extracted_points * 5 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return samples;
}

} // namespace kmeans::clustering
