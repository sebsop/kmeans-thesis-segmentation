/**
 * @file quantum_engine.cu
 * @brief Implementation of the quantum-inspired distance estimation engine.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

/**
 * @brief CUDA kernel for Quantum-Inspired assignment.
 *
 * Instead of standard Euclidean distance, this kernel calculates a similarity
 * metric inspired by the "Swap Test" in quantum computing.
 *
 * @details The similarity between point P and centroid C is modeled as the
 * probability of a quantum state overlap. The distance is defined as:
 *   Dist = 1.0 - Product( cos^2( (P[d] - C[d]) * Scale ) )
 *
 * This mapping allows the clustering to operate in a non-linear phase space,
 * which can be more sensitive to specific feature variations.
 *
 * @param samples Global device pointer to feature matrix.
 * @param numPoints Number of points.
 * @param centers Global device pointer to current centroids cached in shared memory.
 * @param k Number of clusters.
 * @param labels [Output] Cluster indices for each point.
 * @param changed [Output] Flag set to 1 if any point changed its cluster.
 * @param scale_factor Scaling constant to map features to the [-PI, PI] range.
 */
__global__ static void quantumAssignKernel(const float* __restrict__ samples, int numPoints,
                                           const float* __restrict__ centers, int k, int* __restrict__ labels,
                                           int* __restrict__ changed, float scale_factor) {
    extern __shared__ float s_mem[];
    float* s_centers = s_mem;

    // 1. Cooperative Load
    int tid = threadIdx.x;
    if (tid < k * constants::clustering::FEATURE_DIMS) {
        s_centers[tid] = centers[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    // 2. Feature Loading
    float p[constants::clustering::FEATURE_DIMS];
#pragma unroll
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        p[d] = samples[idx * constants::clustering::FEATURE_DIMS + d];
    }

    float minDistSq = constants::math::INF;
    int bestK = 0;

    // 3. Quantum Similarity Calculation
    for (int j = 0; j < k; ++j) {
        float target_prob = 1.0f;
#pragma unroll
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            // Map feature difference to a phase shift
            float diff = (p[d] - s_centers[j * constants::clustering::FEATURE_DIMS + d]) * scale_factor *
                         constants::quantum::PHASE_OFFSET;

            // Use fast hardware intrinsics for cosine
            float cos_val = __cosf(diff);
            target_prob *= (cos_val * cos_val);
        }

        // Distance is the inverse of the overlap probability
        float dist = 1.0f - target_prob;
        if (dist < minDistSq) {
            minDistSq = dist;
            bestK = j;
        }
    }

    // 4. Update and Convergence Check
    if (labels[idx] != bestK) {
        labels[idx] = bestK;
        atomicOr(changed, 1);
    }
}

void QuantumEngine::launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels,
                                           int* d_changed, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    quantumAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samples, numPoints, d_centers, k, d_labels,
                                                                        d_changed, constants::quantum::SCALE_FACTOR);
}

} // namespace kmeans::clustering
