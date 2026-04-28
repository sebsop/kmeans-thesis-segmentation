#include <algorithm>
#include <cmath>
#include <limits>

#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"

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

void QuantumEngine::preRunSetup(const std::vector<cv::Vec<float, 5>>& initialCenters, const cv::Mat& samples) {
    float min_vals[5] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
    float max_vals[5] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};

    if (!samples.empty()) {
        int numPoints = samples.rows;
        for (int i = 0; i < numPoints; ++i) {
            const float* rowPtr = samples.ptr<float>(i);
            for (int d = 0; d < 5; ++d) {
                float val = rowPtr[d];
                if (val < min_vals[d])
                    min_vals[d] = val;
                if (val > max_vals[d])
                    max_vals[d] = val;
            }
        }
    } else {
        // Compute from centers if samples matrix is empty (GPU direct path)
        for (const auto& c : initialCenters) {
            for (int d = 0; d < 5; ++d) {
                if (c[d] < min_vals[d])
                    min_vals[d] = c[d];
                if (c[d] > max_vals[d])
                    max_vals[d] = c[d];
            }
        }
    }

    float max_range = 0.0f;
    for (int d = 0; d < 5; ++d) {
        float range = max_vals[d] - min_vals[d];
        if (range > max_range)
            max_range = range;
    }
    float global_range = max_range + 1e-8f;
    m_scaleFactor = (constants::PI_F / 2.0f) / global_range;
}

void QuantumEngine::launchAssignKernel(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels,
                                       int* d_changed, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    quantumAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samples, numPoints, d_centers, k, d_labels,
                                                                        d_changed, m_scaleFactor);
}

} // namespace kmeans::clustering
