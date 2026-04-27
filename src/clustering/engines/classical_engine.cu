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

__global__ static void classicalUpdateKernel(const float* __restrict__ samples, int numPoints,
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

void ClassicalEngine::launchAssignKernel(float* d_samples, int numPoints, float* d_centers, int k,
                                         int* d_labels, int* d_changed, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    classicalAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(
        d_samples, numPoints, d_centers, k, d_labels, d_changed);
}

void ClassicalEngine::launchUpdateKernel(float* d_samples, int numPoints, int k,
                                         int* d_labels, float* d_newSums, int* d_counts, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    classicalUpdateKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(
        d_samples, numPoints, d_labels, k, d_newSums, d_counts);
}

} // namespace kmeans::clustering
