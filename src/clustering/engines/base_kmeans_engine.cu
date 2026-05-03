#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "clustering/engines/base_kmeans_engine.hpp"
#include "clustering/engines/classical_engine.hpp"
#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"
#include "common/scoped_timer.hpp"
#include "common/utils.hpp"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

__global__ static void internalBaseUpdateKernel(const float* __restrict__ samples, int numPoints,
                                                const int* __restrict__ labels, int k, float* __restrict__ newSums,
                                                int* __restrict__ counts) {
    extern __shared__ float s_mem[];
    float* s_sums = s_mem;
    int* s_counts = (int*)&s_mem[k * constants::FEATURE_DIMS];

    int tid = threadIdx.x;
    int total_elements = k * constants::FEATURE_DIMS + k;
    if (tid < total_elements) {
        s_mem[tid] = 0.0f;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) [[likely]] {
        int cluster = labels[idx];
        if (cluster >= 0 && cluster < k) [[likely]] {
            atomicAdd(&s_counts[cluster], 1);
#pragma unroll
            for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
                atomicAdd(&s_sums[cluster * constants::FEATURE_DIMS + d], samples[idx * constants::FEATURE_DIMS + d]);
            }
        }
    }
    __syncthreads();

    if (tid < k) {
        if (s_counts[tid] > 0) [[likely]] {
            atomicAdd(&counts[tid], s_counts[tid]);
#pragma unroll
            for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
                atomicAdd(&newSums[tid * constants::FEATURE_DIMS + d], s_sums[tid * constants::FEATURE_DIMS + d]);
            }
        }
    }
}

template <typename Derived>
void BaseKMeansEngine<Derived>::baseUpdateKernel(float* d_samp, int numPoints, int k, int* d_lab, float* d_nSums, int* d_cnts,
                                                int threadsPerBlock, int blocksPerGrid, size_t sharedSize) const {
    internalBaseUpdateKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samp, numPoints, d_lab, k, d_nSums,
                                                                             d_cnts);
}

template <typename Derived>
BaseKMeansEngine<Derived>::~BaseKMeansEngine() {
    if (m_d_samples)
        cudaFree(m_d_samples);
    if (m_d_centers)
        cudaFree(m_d_centers);
    if (m_d_labels)
        cudaFree(m_d_labels);
    if (m_d_newSums)
        cudaFree(m_d_newSums);
    if (m_d_counts)
        cudaFree(m_d_counts);
    if (m_d_changed)
        cudaFree(m_d_changed);
}

template <typename Derived>
void BaseKMeansEngine<Derived>::ensureBuffers(int numPoints, int k) {
    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        if (m_d_samples)
            cudaFree(m_d_samples);
        if (m_d_centers)
            cudaFree(m_d_centers);
        if (m_d_labels)
            cudaFree(m_d_labels);
        if (m_d_newSums)
            cudaFree(m_d_newSums);
        if (m_d_counts)
            cudaFree(m_d_counts);
        if (m_d_changed)
            cudaFree(m_d_changed);

        m_maxPoints = std::max(m_maxPoints, static_cast<size_t>(numPoints));
        m_maxK = std::max(m_maxK, k);

        CUDA_CHECK(cudaMalloc(&m_d_samples, m_maxPoints * constants::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_centers, m_maxK * constants::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_labels, m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&m_d_newSums, m_maxK * constants::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_counts, m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&m_d_changed, sizeof(int)));
    }
}

template <typename Derived>
std::vector<FeatureVector>
BaseKMeansEngine<Derived>::run(const cv::Mat& samples,
                      const std::vector<FeatureVector>& initialCenters, int k,
                      int maxIterations) {
    assert(k >= constants::K_MIN && k <= constants::K_MAX);
    int numPoints = samples.rows;
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    ensureBuffers(numPoints, k);
    CUDA_CHECK(cudaMemcpy(m_d_samples, samples.ptr<float>(0), numPoints * constants::FEATURE_DIMS * sizeof(float),
                          cudaMemcpyHostToDevice));

    static_cast<Derived*>(this)->preRunSetupImpl(initialCenters, samples);

    return runInternal(m_d_samples, numPoints, initialCenters, k, maxIterations);
}

template <typename Derived>
std::vector<FeatureVector>
BaseKMeansEngine<Derived>::runOnDevice(float* d_samples_ext, int numPoints,
                              const std::vector<FeatureVector>& initialCenters, int k,
                              int maxIterations) {
    assert(k >= constants::K_MIN && k <= constants::K_MAX);
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        if (m_d_centers)
            cudaFree(m_d_centers);
        if (m_d_labels)
            cudaFree(m_d_labels);
        if (m_d_newSums)
            cudaFree(m_d_newSums);
        if (m_d_counts)
            cudaFree(m_d_counts);
        if (m_d_changed)
            cudaFree(m_d_changed);

        m_maxPoints = std::max(m_maxPoints, static_cast<size_t>(numPoints));
        m_maxK = std::max(m_maxK, k);

        CUDA_CHECK(cudaMalloc(&m_d_centers, m_maxK * constants::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_labels, m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&m_d_newSums, m_maxK * constants::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_counts, m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&m_d_changed, sizeof(int)));
    }

    static_cast<Derived*>(this)->preRunSetupImpl(initialCenters, cv::Mat());

    return runInternal(d_samples_ext, numPoints, initialCenters, k, maxIterations);
}

template <typename Derived>
std::vector<FeatureVector>
BaseKMeansEngine<Derived>::runInternal(float* d_samp, int numPoints,
                              const std::vector<FeatureVector>& initialCenters, int k,
                              int maxIterations) {
    common::ScopedTimer timer("KMeans Execution");
    size_t centersSize = static_cast<size_t>(k) * constants::FEATURE_DIMS * sizeof(float);

    std::vector<int> k_indices(k);
    std::iota(k_indices.begin(), k_indices.end(), 0);
    std::vector<int> d_indices(constants::FEATURE_DIMS);
    std::iota(d_indices.begin(), d_indices.end(), 0);

    std::for_each(k_indices.begin(), k_indices.end(), [&](int i) {
        std::for_each(d_indices.begin(), d_indices.end(), [&](int d) {
            h_centers[i * constants::FEATURE_DIMS + d] = initialCenters[i][d];
        });
    });

    CUDA_CHECK(cudaMemcpy(m_d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(m_d_labels, 0xFF, numPoints * sizeof(int)));

    int threadsPerBlock = constants::CUDA_THREADS_PER_BLOCK;
    int blocksPerGrid = common::calculateGridDim(numPoints, threadsPerBlock);

    size_t sharedAssignSize = static_cast<size_t>(k) * constants::FEATURE_DIMS * sizeof(float);
    size_t sharedUpdateSize =
        static_cast<size_t>(k) * constants::FEATURE_DIMS * sizeof(float) + static_cast<size_t>(k) * sizeof(int);

    std::vector<float> h_newSums(k * constants::FEATURE_DIMS);
    std::vector<int> h_counts(k);

    int iter = 0;
    for (; iter < maxIterations; ++iter) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(m_d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        static_cast<Derived*>(this)->launchAssignKernelImpl(d_samp, numPoints, m_d_centers, k, m_d_labels, m_d_changed, threadsPerBlock, blocksPerGrid,
                           sharedAssignSize);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, m_d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed == 0) [[unlikely]] {
            break; // converged
        }

        CUDA_CHECK(cudaMemset(m_d_newSums, 0, centersSize));
        CUDA_CHECK(cudaMemset(m_d_counts, 0, k * sizeof(int)));

        baseUpdateKernel(d_samp, numPoints, k, m_d_labels, m_d_newSums, m_d_counts, threadsPerBlock, blocksPerGrid,
                         sharedUpdateSize);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_newSums.data(), m_d_newSums, centersSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), m_d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));

        // CPU center averaging
        std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
            if (h_counts[j] > 0) [[likely]] {
                std::for_each(d_indices.begin(), d_indices.end(), [&](int d) {
                    h_centers[j * constants::FEATURE_DIMS + d] =
                        h_newSums[j * constants::FEATURE_DIMS + d] / static_cast<float>(h_counts[j]);
                });
            } else if (numPoints > 0) {
                int randomIdx = rand() % numPoints;
                CUDA_CHECK(cudaMemcpy(&h_centers[j * constants::FEATURE_DIMS],
                                      &d_samp[randomIdx * constants::FEATURE_DIMS],
                                      constants::FEATURE_DIMS * sizeof(float), cudaMemcpyDeviceToHost));
            }
        });
        CUDA_CHECK(cudaMemcpy(m_d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    }

    m_lastIterations = iter + 1;

    std::vector<FeatureVector> finalCenters(k);
    std::for_each(k_indices.begin(), k_indices.end(), [&](int i) {
        std::for_each(d_indices.begin(), d_indices.end(), [&](int d) {
            finalCenters[i][d] = h_centers[i * constants::FEATURE_DIMS + d];
        });
    });
    return finalCenters;
}

template class BaseKMeansEngine<ClassicalEngine>;
template class BaseKMeansEngine<QuantumEngine>;

} // namespace kmeans::clustering
