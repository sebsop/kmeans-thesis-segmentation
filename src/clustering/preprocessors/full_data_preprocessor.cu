#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

#include "clustering/preprocessors/full_data_preprocessor.hpp"
#include "common/constants.hpp"

#define CUDA_CHECK_PREP(call)                                                                                          \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

__global__ void preprocess_features_kernel(const uchar3* __restrict__ frame_data, float* __restrict__ samples, int cols,
                                           int rows, float invCols, float invRows, float color_scale,
                                           float spatial_scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        uchar3 bgr = frame_data[idx];

        float x01 = static_cast<float>(x) * invCols;
        float y01 = static_cast<float>(y) * invRows;

        samples[idx * 5 + 0] = static_cast<float>(bgr.x) * color_scale;
        samples[idx * 5 + 1] = static_cast<float>(bgr.y) * color_scale;
        samples[idx * 5 + 2] = static_cast<float>(bgr.z) * color_scale;
        samples[idx * 5 + 3] = x01 * spatial_scale;
        samples[idx * 5 + 4] = y01 * spatial_scale;
    }
}

cv::Mat FullDataPreprocessor::prepare(const cv::Mat& frame) {
    CV_Assert(frame.type() == CV_8UC3 && frame.isContinuous());

    int n = frame.rows * frame.cols;
    cv::Mat samples(n, 5, CV_32F);

    float invCols = 1.0f / static_cast<float>(frame.cols);
    float invRows = 1.0f / static_cast<float>(frame.rows);

    uchar3* d_frame_data = nullptr;
    float* d_samples = nullptr;

    CUDA_CHECK_PREP(cudaMalloc(&d_frame_data, n * sizeof(uchar3)));
    CUDA_CHECK_PREP(cudaMalloc(&d_samples, n * 5 * sizeof(float)));

    CUDA_CHECK_PREP(cudaMemcpy(d_frame_data, frame.ptr<uchar3>(), n * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((frame.cols + blockSize.x - 1) / blockSize.x, (frame.rows + blockSize.y - 1) / blockSize.y);

    preprocess_features_kernel<<<gridSize, blockSize>>>(d_frame_data, d_samples, frame.cols, frame.rows, invCols,
                                                        invRows, kmeans::constants::COLOR_SCALE,
                                                        kmeans::constants::SPATIAL_SCALE);

    CUDA_CHECK_PREP(cudaDeviceSynchronize());

    CUDA_CHECK_PREP(cudaMemcpy(samples.ptr<float>(), d_samples, n * 5 * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK_PREP(cudaFree(d_frame_data));
    CUDA_CHECK_PREP(cudaFree(d_samples));

    return samples;
}

} // namespace kmeans::clustering
