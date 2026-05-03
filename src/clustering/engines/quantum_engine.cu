#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {

__global__ static void quantumAssignKernel(const float* __restrict__ samples, int numPoints,
                                           const float* __restrict__ centers, int k, int* __restrict__ labels,
                                           int* __restrict__ changed, float scale_factor) {
    extern __shared__ float s_mem[];
    float* s_centers = s_mem;

    int tid = threadIdx.x;
    if (tid < k * constants::clustering::FEATURE_DIMS) {
        s_centers[tid] = centers[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    float p[constants::clustering::FEATURE_DIMS];
#pragma unroll
    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        p[d] = samples[idx * constants::clustering::FEATURE_DIMS + d];
    }

    float minDistSq = constants::math::INF;
    int bestK = 0;

    for (int j = 0; j < k; ++j) {
        float target_prob = 1.0f;
#pragma unroll
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            float diff = (p[d] - s_centers[j * constants::clustering::FEATURE_DIMS + d]) * scale_factor *
                         constants::quantum::PHASE_OFFSET;
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

void QuantumEngine::launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels,
                                           int* d_changed, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    quantumAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samples, numPoints, d_centers, k, d_labels,
                                                                        d_changed, m_scaleFactor);
}

} // namespace kmeans::clustering
