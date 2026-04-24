#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "clustering/engines/classical_engine.hpp"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

__global__ static void classicalAssignKernel(const float* __restrict__ samples, int numPoints,
                                             const float* __restrict__ centers, int k, int* __restrict__ labels,
                                             int* __restrict__ changed) {
    extern __shared__ float s_centers[];

    int tid = threadIdx.x;
    // Max K=20, so k*5 = 100 elements. Our block is usually 256 threads.
    if (tid < k * 5) {
        s_centers[tid] = centers[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    float p[5];
#pragma unroll
    for (int d = 0; d < 5; ++d) {
        p[d] = samples[idx * 5 + d];
    }

    float minDistSq = 1e30f;
    int bestK = 0;

    for (int j = 0; j < k; ++j) {
        float d2 = 0.0f;
#pragma unroll
        for (int d = 0; d < 5; ++d) {
            float diff = p[d] - s_centers[j * 5 + d];
            d2 += diff * diff;
        }
        if (d2 < minDistSq) {
            minDistSq = d2;
            bestK = j;
        }
    }

    if (labels[idx] != bestK) {
        labels[idx] = bestK;
        *changed = 1;
    }
}

__global__ static void classicalUpdateKernel(const float* __restrict__ samples, int numPoints,
                                             const int* __restrict__ labels, int k, float* __restrict__ newSums,
                                             int* __restrict__ counts) {
    extern __shared__ float s_mem[];
    float* s_sums = s_mem;
    int* s_counts = (int*)&s_mem[k * 5];

    int tid = threadIdx.x;
    int total_elements = k * 5 + k;
    if (tid < total_elements) {
        s_mem[tid] = 0.0f; // 0.0f is equivalent to integer 0
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

std::vector<cv::Vec<float, 5>> ClassicalEngine::run(const cv::Mat& samples,
                                                    const std::vector<cv::Vec<float, 5>>& initialCenters, int k) {
    int numPoints = samples.rows;
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    float* d_samples;
    float* d_centers;
    int* d_labels;
    float* d_newSums;
    int* d_counts;
    int* d_changed;

    size_t samplesSize = numPoints * 5 * sizeof(float);
    size_t centersSize = k * 5 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_samples, samplesSize));
    CUDA_CHECK(cudaMalloc(&d_centers, centersSize));
    CUDA_CHECK(cudaMalloc(&d_labels, numPoints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_newSums, centersSize));
    CUDA_CHECK(cudaMalloc(&d_counts, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    std::vector<float> h_centers(k * 5);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < 5; ++d) {
            h_centers[i * 5 + d] = initialCenters[i][d];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_samples, samples.ptr<float>(0), samplesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_labels, 0xFF, numPoints * sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedAssignSize = k * 5 * sizeof(float);
    size_t sharedUpdateSize = k * 5 * sizeof(float) + k * sizeof(int);

    std::vector<float> h_newSums(k * 5);
    std::vector<int> h_counts(k);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    for (int iter = 0; iter < 20; ++iter) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        classicalAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedAssignSize>>>(d_samples, numPoints, d_centers, k,
                                                                                    d_labels, d_changed);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed == 0) {
            break;
        }

        CUDA_CHECK(cudaMemset(d_newSums, 0, centersSize));
        CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(int)));

        classicalUpdateKernel<<<blocksPerGrid, threadsPerBlock, sharedUpdateSize>>>(d_samples, numPoints, d_labels, k,
                                                                                    d_newSums, d_counts);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_newSums.data(), d_newSums, centersSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));

        for (int j = 0; j < k; ++j) {
            if (h_counts[j] > 0) {
                for (int d = 0; d < 5; ++d) {
                    h_centers[j * 5 + d] = h_newSums[j * 5 + d] / static_cast<float>(h_counts[j]);
                }
            } else {
                int rand_idx = dis(gen);
                for (int d = 0; d < 5; ++d) {
                    h_centers[j * 5 + d] = samples.at<float>(rand_idx, d);
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    }

    std::vector<cv::Vec<float, 5>> finalCenters(k);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < 5; ++d) {
            finalCenters[i][d] = h_centers[i * 5 + d];
        }
    }

    cudaFree(d_samples);
    cudaFree(d_centers);
    cudaFree(d_labels);
    cudaFree(d_newSums);
    cudaFree(d_counts);
    cudaFree(d_changed);

    return finalCenters;
}

} // namespace kmeans::clustering
