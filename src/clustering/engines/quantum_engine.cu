#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "clustering/engines/quantum_engine.hpp"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

__global__ static void quantumAssignKernel(const float* __restrict__ samples, int numPoints,
                                           const float* __restrict__ centers, int k, int* __restrict__ labels,
                                           int* __restrict__ changed, float scale_factor) {
    extern __shared__ float s_centers[];

    int tid = threadIdx.x;
    if (tid < k * 5) {
        s_centers[tid] = centers[tid] * scale_factor; // Pre-scaled
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    float p[5];
#pragma unroll
    for (int d = 0; d < 5; ++d) {
        p[d] = samples[idx * 5 + d] * scale_factor; // Pre-scaled
    }

    float minDistSq = 1e30f;
    int bestK = 0;

    for (int j = 0; j < k; ++j) {
        float target_prob = 1.0f;
#pragma unroll
        for (int d = 0; d < 5; ++d) {
            // Eliminated 2 multiplications per dimension!
            float diff = (p[d] - s_centers[j * 5 + d]) * 0.5f;
            float cos_val = __cosf(diff);
            target_prob *= (cos_val * cos_val);
        }
        float dist = 1.0f - target_prob;
        if (dist < minDistSq) {
            minDistSq = dist;
            bestK = j;
        }
    }

    if (labels[idx] != bestK) {
        labels[idx] = bestK;
        atomicOr(changed, 1);
    }
}

__global__ static void quantumUpdateKernel(const float* __restrict__ samples, int numPoints,
                                           const int* __restrict__ labels, int k, float* __restrict__ newSums,
                                           int* __restrict__ counts) {
    extern __shared__ float s_mem[];
    float* s_sums = s_mem;
    int* s_counts = (int*)&s_mem[k * 5];

    int tid = threadIdx.x;
    int total_elements = k * 5 + k;
    if (tid < total_elements) {
        s_mem[tid] = 0.0f;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int cluster = labels[idx];
        if (cluster >= 0 && cluster < k) {
            atomicAdd(&s_counts[cluster], 1);
#pragma unroll
            for (int d = 0; d < 5; ++d) {
                atomicAdd(&s_sums[cluster * 5 + d], samples[idx * 5 + d]);
            }
        }
    }
    __syncthreads();

    if (tid < k) {
        if (s_counts[tid] > 0) {
            atomicAdd(&counts[tid], s_counts[tid]);
#pragma unroll
            for (int d = 0; d < 5; ++d) {
                atomicAdd(&newSums[tid * 5 + d], s_sums[tid * 5 + d]);
            }
        }
    }
}

QuantumEngine::~QuantumEngine() {
    if (d_samples)
        cudaFree(d_samples);
    if (d_centers)
        cudaFree(d_centers);
    if (d_labels)
        cudaFree(d_labels);
    if (d_newSums)
        cudaFree(d_newSums);
    if (d_counts)
        cudaFree(d_counts);
    if (d_changed)
        cudaFree(d_changed);
}

void QuantumEngine::ensureBuffers(int numPoints, int k) {
    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        if (d_samples)  cudaFree(d_samples);
        if (d_centers)  cudaFree(d_centers);
        if (d_labels)   cudaFree(d_labels);
        if (d_newSums)  cudaFree(d_newSums);
        if (d_counts)   cudaFree(d_counts);
        if (d_changed)  cudaFree(d_changed);

        m_maxPoints = std::max(m_maxPoints, static_cast<size_t>(numPoints));
        m_maxK      = std::max(m_maxK, k);

        CUDA_CHECK(cudaMalloc(&d_samples, m_maxPoints * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centers, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels,  m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_newSums, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts,  m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    }
}

std::vector<cv::Vec<float, 5>> QuantumEngine::runInternal(float* d_samp, int numPoints,
                                                           const std::vector<cv::Vec<float, 5>>& initialCenters,
                                                           int k, float scale_factor) {
    size_t centersSize = static_cast<size_t>(k) * 5 * sizeof(float);

    std::vector<float> h_centers(k * 5);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < 5; ++d) {
            h_centers[i * 5 + d] = initialCenters[i][d];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_labels, 0xFF, numPoints * sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid   = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedAssignSize = static_cast<size_t>(k) * 5 * sizeof(float);
    size_t sharedUpdateSize = static_cast<size_t>(k) * 5 * sizeof(float) + static_cast<size_t>(k) * sizeof(int);

    std::vector<float> h_newSums(k * 5);
    std::vector<int>   h_counts(k);

    int iter = 0;
    for (; iter < 20; ++iter) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        quantumAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedAssignSize>>>(
            d_samp, numPoints, d_centers, k, d_labels, d_changed, scale_factor);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed == 0) {
            break;
        }

        CUDA_CHECK(cudaMemset(d_newSums, 0, centersSize));
        CUDA_CHECK(cudaMemset(d_counts,  0, k * sizeof(int)));

        quantumUpdateKernel<<<blocksPerGrid, threadsPerBlock, sharedUpdateSize>>>(
            d_samp, numPoints, d_labels, k, d_newSums, d_counts);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_newSums.data(), d_newSums, centersSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(),  d_counts,  k * sizeof(int), cudaMemcpyDeviceToHost));

        for (int j = 0; j < k; ++j) {
            if (h_counts[j] > 0) {
                for (int d = 0; d < 5; ++d) {
                    h_centers[j * 5 + d] = h_newSums[j * 5 + d] / static_cast<float>(h_counts[j]);
                }
            }
            // Empty cluster: keep existing center — rare at steady state with warm starts
        }
        CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    }

    m_lastIterations = iter + 1;

    std::vector<cv::Vec<float, 5>> finalCenters(k);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < 5; ++d) {
            finalCenters[i][d] = h_centers[i * 5 + d];
        }
    }
    return finalCenters;
}

static float computeScaleFactor(const cv::Mat& samples) {
    int numPoints = samples.rows;
    float min_vals[5] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
    float max_vals[5] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};

    for (int i = 0; i < numPoints; ++i) {
        const float* rowPtr = samples.ptr<float>(i);
        for (int d = 0; d < 5; ++d) {
            float val = rowPtr[d];
            if (val < min_vals[d]) min_vals[d] = val;
            if (val > max_vals[d]) max_vals[d] = val;
        }
    }

    float max_range = 0.0f;
    for (int d = 0; d < 5; ++d) {
        float range = max_vals[d] - min_vals[d];
        if (range > max_range) max_range = range;
    }
    float global_range = max_range + 1e-8f;
    return (static_cast<float>(CV_PI) / 2.0f) / global_range;
}

std::vector<cv::Vec<float, 5>> QuantumEngine::run(const cv::Mat& samples,
                                                   const std::vector<cv::Vec<float, 5>>& initialCenters, int k) {
    int numPoints = samples.rows;
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    float scale_factor = computeScaleFactor(samples);

    ensureBuffers(numPoints, k);
    CUDA_CHECK(cudaMemcpy(d_samples, samples.ptr<float>(0), numPoints * 5 * sizeof(float), cudaMemcpyHostToDevice));

    return runInternal(d_samples, numPoints, initialCenters, k, scale_factor);
}

std::vector<cv::Vec<float, 5>> QuantumEngine::runOnDevice(float* d_samples_ext, int numPoints,
                                                           const std::vector<cv::Vec<float, 5>>& initialCenters,
                                                           int k) {
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    float min_vals[5] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
    float max_vals[5] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};
    for (const auto& c : initialCenters) {
        for (int d = 0; d < 5; ++d) {
            if (c[d] < min_vals[d]) min_vals[d] = c[d];
            if (c[d] > max_vals[d]) max_vals[d] = c[d];
        }
    }
    float max_range = 0.0f;
    for (int d = 0; d < 5; ++d) {
        float range = max_vals[d] - min_vals[d];
        if (range > max_range) max_range = range;
    }
    float scale_factor = (static_cast<float>(CV_PI) / 2.0f) / (max_range + 1e-8f);

    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        if (d_centers)  cudaFree(d_centers);
        if (d_labels)   cudaFree(d_labels);
        if (d_newSums)  cudaFree(d_newSums);
        if (d_counts)   cudaFree(d_counts);
        if (d_changed)  cudaFree(d_changed);

        m_maxPoints = std::max(m_maxPoints, static_cast<size_t>(numPoints));
        m_maxK      = std::max(m_maxK, k);

        CUDA_CHECK(cudaMalloc(&d_centers, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels,  m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_newSums, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts,  m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    }

    return runInternal(d_samples_ext, numPoints, initialCenters, k, scale_factor);
}

} // namespace kmeans::clustering
