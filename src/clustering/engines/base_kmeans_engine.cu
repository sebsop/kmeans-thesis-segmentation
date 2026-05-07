/**
 * @file base_kmeans_engine.cu
 * @brief Base implementation and shared kernels for the K-Means pipeline.
 */

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "clustering/engines/base_kmeans_engine.hpp"
#include "clustering/engines/classical_engine.hpp"
#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"
#include "common/utils.hpp"

namespace {
/** @brief Local helper to handle CUDA errors with file/line information. */
inline void cudaCheck(cudaError_t err, const std::source_location loc = std::source_location::current()) {
    if (err != cudaSuccess) {
        std::string msg = std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + loc.file_name() + ":" +
                          std::to_string(loc.line());
        throw std::runtime_error(msg);
    }
}
} // namespace

#define CUDA_CHECK(call) cudaCheck(call)

namespace kmeans::clustering {

/**
 * @brief Shared CUDA kernel for the "Update" step of K-Means.
 *
 * This kernel aggregates the feature vectors assigned to each cluster to compute
 * new cluster centers.
 *
 * @note OPTIMIZATION: Uses a two-tier aggregation strategy:
 *       1. Partial sums are computed in __shared__ memory using atomicAdd.
 *          This drastically reduces contention on global memory.
 *       2. Block-level partial sums are then added to the global buffers
 *          (newSums, counts) using atomicAdd.
 *
 * @param samples Global device pointer to 5D feature matrix.
 * @param numPoints Number of rows in the matrix.
 * @param labels Cluster assignments for each point.
 * @param k Number of clusters.
 * @param newSums [Output] Aggregated sums of features per cluster.
 * @param counts [Output] Number of points assigned to each cluster.
 */
__global__ static void internalBaseUpdateKernel(const float* __restrict__ samples, int numPoints,
                                                const int* __restrict__ labels, int k, float* __restrict__ newSums,
                                                int* __restrict__ counts) {
    extern __shared__ float s_mem[];
    float* s_sums = s_mem;
    int* s_counts = reinterpret_cast<int*>(&s_mem[k * constants::clustering::FEATURE_DIMS]);

    int tid = threadIdx.x;
    int total_elements = k * constants::clustering::FEATURE_DIMS + k;

    // 1. Initialize shared memory to zero
    if (tid < total_elements) {
        s_mem[tid] = 0.0f;
    }
    __syncthreads();

    // 2. Aggregate local points into block-level partial sums
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) [[likely]] {
        int cluster = labels[idx];
        if (cluster >= 0 && cluster < k) [[likely]] {
            atomicAdd(&s_counts[cluster], 1);
#pragma unroll
            for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
                atomicAdd(&s_sums[cluster * constants::clustering::FEATURE_DIMS + d],
                          samples[idx * constants::clustering::FEATURE_DIMS + d]);
            }
        }
    }
    __syncthreads();

    // 3. Aggregate block-level sums into global memory
    if (tid < k) {
        if (s_counts[tid] > 0) [[likely]] {
            atomicAdd(&counts[tid], s_counts[tid]);
#pragma unroll
            for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
                atomicAdd(&newSums[tid * constants::clustering::FEATURE_DIMS + d],
                          s_sums[tid * constants::clustering::FEATURE_DIMS + d]);
            }
        }
    }
}

template <typename Derived>
void BaseKMeansEngine<Derived>::baseUpdateKernel(float* d_samples, int numPoints, int k, int* d_labels,
                                                 float* d_newSums, int* d_counts, int threadsPerBlock,
                                                 int blocksPerGrid, size_t sharedSize) const {
    internalBaseUpdateKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samples, numPoints, d_labels, k,
                                                                             d_newSums, d_counts);
}

template <typename Derived> BaseKMeansEngine<Derived>::~BaseKMeansEngine() {
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

/**
 * @brief Ensures that GPU buffers are large enough for the current task.
 *
 * To maximize performance, this method only reallocates if the requested
 * size exceeds the current capacity, preventing fragmentation during
 * dynamic cluster size changes.
 */
template <typename Derived> void BaseKMeansEngine<Derived>::ensureBuffers(int numPoints, int k) {
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

        CUDA_CHECK(cudaMalloc(&m_d_samples, m_maxPoints * constants::clustering::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_centers, m_maxK * constants::clustering::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_labels, m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&m_d_newSums, m_maxK * constants::clustering::FEATURE_DIMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_counts, m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&m_d_changed, sizeof(int)));
    }
}

/**
 * @brief Main entry point for clustering when data is on the Host (CPU).
 *
 * This method handles the data transfer from an OpenCV Mat to the GPU
 * before starting the K-Means iterations.
 *
 * @param samples Input feature matrix (OpenCV Mat).
 * @param initialCenters Starting positions for centroids.
 * @param k Number of clusters.
 * @param maxIterations Safety cutoff for the algorithm.
 * @return std::vector<FeatureVector> Final calculated centroids.
 */
template <typename Derived>
std::vector<FeatureVector> BaseKMeansEngine<Derived>::run(const cv::Mat& samples,
                                                          std::span<const FeatureVector> initialCenters, int k,
                                                          int maxIterations) {
    assert(k >= constants::clustering::K_MIN && k <= constants::clustering::K_MAX);
    int numPoints = samples.rows;
    if (numPoints == 0 || k <= 0)
        return std::vector<FeatureVector>(initialCenters.begin(), initialCenters.end());

    ensureBuffers(numPoints, k);
    CUDA_CHECK(cudaMemcpy(m_d_samples, samples.ptr<float>(0),
                          numPoints * constants::clustering::FEATURE_DIMS * sizeof(float), cudaMemcpyHostToDevice));

    return runInternal(m_d_samples, numPoints, initialCenters, k, maxIterations);
}

/**
 * @brief High-performance entry point for data already on the Device (GPU).
 *
 * This method enables a "Zero-Copy" workflow by accepting a pointer to VRAM
 * directly, avoiding the expensive Host-to-Device transfer.
 *
 * @param d_samples_ext Pointer to the 5D feature matrix in GPU memory.
 * @param numPoints Total points to process.
 * @param initialCenters Starting positions for centroids.
 * @param k Number of clusters.
 * @param maxIterations Safety cutoff for the algorithm.
 * @return std::vector<FeatureVector> Final calculated centroids.
 */
template <typename Derived>
std::vector<FeatureVector> BaseKMeansEngine<Derived>::runOnDevice(float* d_samples_ext, int numPoints,
                                                                  std::span<const FeatureVector> initialCenters, int k,
                                                                  int maxIterations) {
    assert(k >= constants::clustering::K_MIN && k <= constants::clustering::K_MAX);
    if (numPoints == 0 || k <= 0)
        return std::vector<FeatureVector>(initialCenters.begin(), initialCenters.end());

    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        ensureBuffers(numPoints, k);
    }

    return runInternal(d_samples_ext, numPoints, initialCenters, k, maxIterations);
}

/**
 * @brief The internal execution loop implementing Lloyd's Algorithm.
 *
 * This method coordinates the GPU assignment and update kernels in an
 * iterative loop until either maxIterations is reached or the labels
 * stop changing (convergence).
 *
 * @param d_samples Device pointer to input data.
 * @param numPoints Total points in data.
 * @param initialCenters Starting centroids.
 * @param k Number of clusters.
 * @param maxIterations Safety limit on loop count.
 * @return std::vector<FeatureVector> The final calculated centroids.
 */
template <typename Derived>
std::vector<FeatureVector> BaseKMeansEngine<Derived>::runInternal(float* d_samples, int numPoints,
                                                                  std::span<const FeatureVector> initialCenters, int k,
                                                                  int maxIterations) {
    size_t centersSize = static_cast<size_t>(k) * constants::clustering::FEATURE_DIMS * sizeof(float);
    std::vector<float> h_centers(k * constants::clustering::FEATURE_DIMS);

    std::vector<int> k_indices(k);
    std::iota(k_indices.begin(), k_indices.end(), 0);
    std::vector<int> dim_indices(constants::clustering::FEATURE_DIMS);
    std::iota(dim_indices.begin(), dim_indices.end(), 0);

    // Initial upload of centers
    std::for_each(k_indices.begin(), k_indices.end(), [&](int i) {
        std::for_each(dim_indices.begin(), dim_indices.end(),
                      [&](int d) { h_centers[i * constants::clustering::FEATURE_DIMS + d] = initialCenters[i][d]; });
    });

    CUDA_CHECK(cudaMemcpy(m_d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(m_d_labels, 0xFF, numPoints * sizeof(int)));

    int threadsPerBlock = constants::cuda::THREADS_PER_BLOCK;
    int blocksPerGrid = common::calculateGridDim(numPoints, threadsPerBlock);

    size_t sharedAssignSize = static_cast<size_t>(k) * constants::clustering::FEATURE_DIMS * sizeof(float);
    size_t sharedUpdateSize = static_cast<size_t>(k) * constants::clustering::FEATURE_DIMS * sizeof(float) +
                              static_cast<size_t>(k) * sizeof(int);

    std::vector<float> h_newSums(k * constants::clustering::FEATURE_DIMS);
    std::vector<int> h_counts(k);

    int iter = 0;
    for (; iter < maxIterations; ++iter) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(m_d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        // 1. Assignment Step (Polymorphic via CRTP)
        static_cast<Derived*>(this)->launchAssignKernelImpl(d_samples, numPoints, m_d_centers, k, m_d_labels,
                                                            m_d_changed, threadsPerBlock, blocksPerGrid,
                                                            sharedAssignSize);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check for convergence
        CUDA_CHECK(cudaMemcpy(&h_changed, m_d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed == 0) [[unlikely]] {
            break;
        }

        // 2. Update Step (Shared Kernel)
        CUDA_CHECK(cudaMemset(m_d_newSums, 0, centersSize));
        CUDA_CHECK(cudaMemset(m_d_counts, 0, k * sizeof(int)));

        baseUpdateKernel(d_samples, numPoints, k, m_d_labels, m_d_newSums, m_d_counts, threadsPerBlock, blocksPerGrid,
                         sharedUpdateSize);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Averaging Step (CPU)
        CUDA_CHECK(cudaMemcpy(h_newSums.data(), m_d_newSums, centersSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), m_d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));

        std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
            if (h_counts[j] > 0) [[likely]] {
                std::for_each(dim_indices.begin(), dim_indices.end(), [&](int d) {
                    h_centers[j * constants::clustering::FEATURE_DIMS + d] =
                        h_newSums[j * constants::clustering::FEATURE_DIMS + d] / static_cast<float>(h_counts[j]);
                });
            } else if (numPoints > 0) {
                // ROBUSTNESS: If a cluster becomes empty, re-seed it with a random point
                int randomIdx = rand() % numPoints;
                CUDA_CHECK(cudaMemcpy(&h_centers[j * constants::clustering::FEATURE_DIMS],
                                      &d_samples[randomIdx * constants::clustering::FEATURE_DIMS],
                                      constants::clustering::FEATURE_DIMS * sizeof(float), cudaMemcpyDeviceToHost));
            }
        });
        CUDA_CHECK(cudaMemcpy(m_d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    }

    m_lastIterations = iter + 1;

    // Convert back to FeatureVector format
    std::vector<FeatureVector> finalCenters(k);
    std::for_each(k_indices.begin(), k_indices.end(), [&](int i) {
        std::for_each(dim_indices.begin(), dim_indices.end(),
                      [&](int d) { finalCenters[i][d] = h_centers[i * constants::clustering::FEATURE_DIMS + d]; });
    });
    return finalCenters;
}

// Explicit instantiations for the supported engines
template class BaseKMeansEngine<ClassicalEngine>;
template class BaseKMeansEngine<QuantumEngine>;

} // namespace kmeans::clustering
