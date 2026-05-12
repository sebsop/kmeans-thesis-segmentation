/**
 * @file cuda_kernels.cu
 * @brief Implementation of high-performance GPU pixel assignment.
 */

#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "backend/cuda_assignment_context.hpp"
#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common/vector_math.hpp"

/** @brief Macro for checking CUDA API return codes and throwing on failure. */
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::backend {

/**
 * @brief Constructor pre-allocates all buffers to avoid runtime latency.
 *
 * Uses pinned host memory (cudaMallocHost) to enable high-speed DMA transfers
 * and persistent device memory to minimize the cost of the processing loop.
 */
CudaAssignmentContext::CudaAssignmentContext(int width, int height, int k) : m_width(width), m_height(height), m_k(k) {
    m_imgSize = width * height * 3 * sizeof(unsigned char);
    m_centersSize = k * constants::clustering::FEATURE_DIMS * sizeof(float);

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

/** @brief Destructor ensures all GPU and pinned memory is released. */
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

/**
 * @brief The core CUDA kernel for pixel classification.
 *
 * Each thread processes a single pixel. It computes the Euclidean distance
 * between the pixel's 5D feature and all cluster centroids, assigning
 * the pixel the color of its nearest neighbor.
 *
 * @note SHARED MEMORY OPTIMIZATION: Centroids are loaded into __shared__ memory
 *       cooperatively by all threads in a block. This drastically reduces the
 *       number of global memory reads from O(Pixels * K) to O(Pixels + K).
 *
 * @param input Raw BGR frame data on GPU.
 * @param output Segmented BGR frame data on GPU.
 * @param width Image width.
 * @param height Image height.
 * @param centers Pointer to centroids in global memory.
 * @param k Number of clusters.
 * @param color_scale Normalization factor for RGB values.
 * @param spatial_scale Normalization factor for XY coordinates.
 */
__global__ static void assignPixelsKernel(const unsigned char* input, unsigned char* output, int width, int height,
                                          const float* centers, int k, float color_scale, float spatial_scale) {
    extern __shared__ float s_centers[];

    // 1. Cooperative Load: Threads work together to fill shared memory
    int tid = threadIdx.x;
    int centersCount = k * constants::clustering::FEATURE_DIMS;
    for (int i = tid; i < centersCount; i += blockDim.x) {
        s_centers[i] = centers[i];
    }
    __syncthreads();

    // 2. Identification: Determine which pixel this thread is responsible for
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total)
        return;

    int r = idx / width;
    int c = idx % width;
    int offset = idx * 3;

    // 3. Normalization: Map pixel coordinates to [0, 1] range
    float x01 = static_cast<float>(c) / static_cast<float>(width);
    float y01 = static_cast<float>(r) / static_cast<float>(height);

    // 4. Feature Construction: [B, G, R, X, Y]
    float f[constants::clustering::FEATURE_DIMS] = {
        static_cast<float>(input[offset + 0]) * color_scale, static_cast<float>(input[offset + 1]) * color_scale,
        static_cast<float>(input[offset + 2]) * color_scale, x01 * spatial_scale, y01 * spatial_scale};

    // 5. Classification: Nearest Neighbor search using Shared Memory centers
    int bestIdx = 0;
    float bestDist2 = constants::math::INF;

    for (int ci = 0; ci < k; ++ci) {
        float d2 = common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(
            f, &s_centers[ci * constants::clustering::FEATURE_DIMS]);
        if (d2 < bestDist2) {
            bestDist2 = d2;
            bestIdx = ci;
        }
    }

    // 6. Assignment: Write the winning cluster's color back to the output buffer
    float inv_scale = 1.0f / fmaxf(constants::math::EPSILON, color_scale);
    output[offset + 0] = static_cast<unsigned char>(
        fminf(255.0f, roundf(s_centers[bestIdx * constants::clustering::FEATURE_DIMS + 0] * inv_scale)));
    output[offset + 1] = static_cast<unsigned char>(
        fminf(255.0f, roundf(s_centers[bestIdx * constants::clustering::FEATURE_DIMS + 1] * inv_scale)));
    output[offset + 2] = static_cast<unsigned char>(
        fminf(255.0f, roundf(s_centers[bestIdx * constants::clustering::FEATURE_DIMS + 2] * inv_scale)));
}

/**
 * @brief Orchestrates the end-to-end GPU assignment pipeline.
 */
void CudaAssignmentContext::run(const cv::Mat& frame, const std::vector<FeatureVector>& centers, cv::Mat& output) {
    if (frame.empty() || centers.empty() || frame.cols != m_width || frame.rows != m_height) {
        return;
    }

    if (output.rows != m_height || output.cols != m_width || output.type() != frame.type()) {
        output.create(m_height, m_width, frame.type());
    }

    // 1. Upload: CPU Mat -> Pinned Buffer -> Device VRAM
    std::memcpy(m_h_input_pinned, frame.data, m_imgSize);
    CUDA_CHECK(cudaMemcpyAsync(m_d_input, m_h_input_pinned, m_imgSize, cudaMemcpyHostToDevice, m_stream));

    std::memcpy(m_h_centers_pinned, centers.data(), m_centersSize);
    CUDA_CHECK(cudaMemcpyAsync(m_d_centers, m_h_centers_pinned, m_centersSize, cudaMemcpyHostToDevice, m_stream));

    // 2. Launch: Asynchronous kernel execution
    int threadsPerBlock = constants::cuda::THREADS_PER_BLOCK;
    int blocksPerGrid = common::calculateGridDim(m_width * m_height, threadsPerBlock);
    size_t sharedSize = static_cast<size_t>(m_k) * constants::clustering::FEATURE_DIMS * sizeof(float);

    assignPixelsKernel<<<blocksPerGrid, threadsPerBlock, sharedSize, m_stream>>>(
        m_d_input, m_d_output, m_width, m_height, m_d_centers, m_k, constants::video::COLOR_SCALE,
        constants::video::SPATIAL_SCALE);
    CUDA_CHECK(cudaPeekAtLastError());

    // 3. Download: Device VRAM -> Pinned Buffer -> CPU Mat
    CUDA_CHECK(cudaMemcpyAsync(m_h_output_pinned, m_d_output, m_imgSize, cudaMemcpyDeviceToHost, m_stream));

    // 4. Sync: Wait for non-blocking operations to finalize
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    std::memcpy(output.data, m_h_output_pinned, m_imgSize);
}

} // namespace kmeans::backend
