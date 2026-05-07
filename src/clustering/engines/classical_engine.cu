/**
 * @file classical_engine.cu
 * @brief Implementation of the standard Euclidean K-Means assignment engine.
 */

#include "clustering/engines/classical_engine.hpp"
#include "common/constants.hpp"
#include "common/vector_math.hpp"

namespace kmeans::clustering {

/**
 * @brief CUDA kernel for the Classical (Euclidean) assignment step.
 *
 * Each thread calculates the Euclidean distance between one point and all
 * cluster centroids, assigning the point to the nearest one.
 *
 * @note OPTIMIZATION:
 *       1. Centroids are loaded into __shared__ memory at the start of the block.
 *       2. If a point's assignment changes, an atomicOr is performed on the
 *          global 'changed' flag to signal that another iteration is needed.
 *
 * @param samples Global device pointer to feature matrix.
 * @param numPoints Number of points.
 * @param centers Global device pointer to current centroids.
 * @param k Number of clusters.
 * @param labels [Output] Cluster indices for each point.
 * @param changed [Output] Flag set to 1 if any point changed its cluster.
 */
__global__ static void classicalAssignKernel(const float* __restrict__ samples, int numPoints,
                                             const float* __restrict__ centers, int k, int* __restrict__ labels,
                                             int* __restrict__ changed) {
    extern __shared__ float s_centers[];

    // 1. Cooperative Load: Cache centroids in shared memory
    int tid = threadIdx.x;
    if (tid < k * constants::clustering::FEATURE_DIMS) {
        s_centers[tid] = centers[tid];
    }
    __syncthreads();

    // 2. Nearest Neighbor Search
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) [[likely]] {
        float p[constants::clustering::FEATURE_DIMS];
#pragma unroll
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            p[d] = samples[idx * constants::clustering::FEATURE_DIMS + d];
        }

        float minDistSq = constants::math::INF;
        int bestK = 0;

        for (int j = 0; j < k; ++j) {
            float d2 = common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(
                p, &s_centers[j * constants::clustering::FEATURE_DIMS]);
            if (d2 < minDistSq) {
                minDistSq = d2;
                bestK = j;
            }
        }

        // 3. Convergence Detection
        if (labels[idx] != bestK) {
            labels[idx] = bestK;
            atomicOr(changed, 1);
        }
    }
}

void ClassicalEngine::launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels,
                                             int* d_changed, int threadsPerBlock, int blocksPerGrid,
                                             size_t sharedSize) {
    classicalAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samples, numPoints, d_centers, k, d_labels,
                                                                          d_changed);
}

} // namespace kmeans::clustering
