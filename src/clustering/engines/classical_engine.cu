#include "clustering/engines/classical_engine.hpp"

namespace kmeans::clustering {

__global__ static void classicalAssignKernel(const float* __restrict__ samples, int numPoints,
                                             const float* __restrict__ centers, int k, int* __restrict__ labels,
                                             int* __restrict__ changed) {
    extern __shared__ float s_centers[];

    int tid = threadIdx.x;
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
        atomicOr(changed, 1);
    }
}

void ClassicalEngine::launchAssignKernel(float* d_samples, int numPoints, float* d_centers, int k,
                                         int* d_labels, int* d_changed, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    classicalAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(
        d_samples, numPoints, d_centers, k, d_labels, d_changed);
}

} // namespace kmeans::clustering
