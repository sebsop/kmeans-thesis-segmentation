#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "backend/cuda_assignment_context.hpp"
#include "common/constants.hpp"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::backend {

CudaAssignmentContext::CudaAssignmentContext(int width, int height, int k) : m_width(width), m_height(height), m_k(k) {
    m_imgSize = width * height * 3 * sizeof(unsigned char);
    m_centersSize = k * 5 * sizeof(float);

    // Persistent device memory allocation safely encapsulated
    CUDA_CHECK(cudaMalloc(&m_d_input, m_imgSize));
    CUDA_CHECK(cudaMalloc(&m_d_output, m_imgSize));
    CUDA_CHECK(cudaMalloc(&m_d_centers, m_centersSize));

    // Persistent pinned host memory allocation for zero-copy staging
    CUDA_CHECK(cudaMallocHost(&m_h_input_pinned, m_imgSize));
    CUDA_CHECK(cudaMallocHost(&m_h_output_pinned, m_imgSize));
    CUDA_CHECK(cudaMallocHost(&m_h_centers_pinned, m_centersSize));

    // Initialize the stream for async operations
    CUDA_CHECK(cudaStreamCreate(&m_stream));
}

CudaAssignmentContext::~CudaAssignmentContext() noexcept {
    if (m_d_input)
        cudaFree(m_d_input);
    if (m_d_output)
        cudaFree(m_d_output);
    if (m_d_centers)
        cudaFree(m_d_centers);

    if (m_h_input_pinned)
        cudaFreeHost(m_h_input_pinned);
    if (m_h_output_pinned)
        cudaFreeHost(m_h_output_pinned);
    if (m_h_centers_pinned)
        cudaFreeHost(m_h_centers_pinned);

    if (m_stream)
        cudaStreamDestroy(m_stream);
}

__global__ static void assignPixelsKernel(const unsigned char* input, unsigned char* output, int width, int height,
                                          const float* centers, int k, float color_scale, float spatial_scale) {
    extern __shared__ float s_centers[];

    // Load centers cooperatively into shared memory (100 floats at k=20, trivially fits)
    int tid = threadIdx.x;
    int centersCount = k * 5;
    for (int i = tid; i < centersCount; i += blockDim.x) {
        s_centers[i] = centers[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total)
        return;

    int r = idx / width;
    int c = idx % width;
    int offset = idx * 3;

    float x01 = static_cast<float>(c) / static_cast<float>(width);
    float y01 = static_cast<float>(r) / static_cast<float>(height);

    float f[5] = {static_cast<float>(input[offset + 0]) * color_scale, static_cast<float>(input[offset + 1]) * color_scale,
                  static_cast<float>(input[offset + 2]) * color_scale, x01 * spatial_scale, y01 * spatial_scale};

    int bestIdx = 0;
    float bestDist2 = 1e20f;

    for (int ci = 0; ci < k; ++ci) {
        float d2 = 0.0f;
        for (int d = 0; d < 5; ++d) {
            float diff = f[d] - s_centers[ci * 5 + d]; // shared memory read
            d2 += diff * diff;
        }
        if (d2 < bestDist2) {
            bestDist2 = d2;
            bestIdx = ci;
        }
    }

    float inv_scale = 1.0f / fmaxf(1e-6f, color_scale);
    output[offset + 0] = static_cast<unsigned char>(fminf(255.0f, s_centers[bestIdx * 5 + 0] * inv_scale));
    output[offset + 1] = static_cast<unsigned char>(fminf(255.0f, s_centers[bestIdx * 5 + 1] * inv_scale));
    output[offset + 2] = static_cast<unsigned char>(fminf(255.0f, s_centers[bestIdx * 5 + 2] * inv_scale));
}

void CudaAssignmentContext::run(const cv::Mat& frame, const std::vector<cv::Vec<float, 5>>& centers, cv::Mat& output) {
    // 1. Quick CPU copy to pinned memory to bypass driver staging overhead
    std::memcpy(m_h_input_pinned, frame.data, m_imgSize);
    CUDA_CHECK(cudaMemcpyAsync(m_d_input, m_h_input_pinned, m_imgSize, cudaMemcpyHostToDevice, m_stream));

    // Flatten centers directly into pinned host buffer
    int k_idx = 0;
    for (const auto& c : centers) {
        for (int i = 0; i < 5; ++i) {
            m_h_centers_pinned[k_idx++] = c[i];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(m_d_centers, m_h_centers_pinned, m_centersSize, cudaMemcpyHostToDevice, m_stream));

    // 2. Launch Kernel on Stream
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_width * m_height + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedSize = static_cast<size_t>(m_k) * 5 * sizeof(float);

    assignPixelsKernel<<<blocksPerGrid, threadsPerBlock, sharedSize, m_stream>>>(
        m_d_input, m_d_output, m_width, m_height, m_d_centers, m_k, constants::COLOR_SCALE, constants::SPATIAL_SCALE);
    CUDA_CHECK(cudaPeekAtLastError());

    // 3. Async Download to pinned memory
    CUDA_CHECK(cudaMemcpyAsync(m_h_output_pinned, m_d_output, m_imgSize, cudaMemcpyDeviceToHost, m_stream));

    // 4. Synchronize the stream to wait for transfers to finish
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // 5. Unpack result
    std::memcpy(output.data, m_h_output_pinned, m_imgSize);
}

} // namespace kmeans::backend