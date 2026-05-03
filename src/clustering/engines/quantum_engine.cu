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
    extern __shared__ float s_centers[];

    int tid = threadIdx.x;
    if (tid < k * constants::FEATURE_DIMS) {
        s_centers[tid] = centers[tid] * scale_factor; // Pre-scaled
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    float p[constants::FEATURE_DIMS];
#pragma unroll
    for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
        p[d] = samples[idx * constants::FEATURE_DIMS + d] * scale_factor; // Pre-scaled
    }

    float minDistSq = constants::MATH_INF;
    int bestK = 0;

    for (int j = 0; j < k; ++j) {
        float target_prob = 1.0f;
#pragma unroll
        for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
            float diff = (p[d] - s_centers[j * constants::FEATURE_DIMS + d]) * constants::QUANTUM_PHASE_OFFSET;
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

void QuantumEngine::preRunSetupImpl(std::span<const FeatureVector> initialCenters,
                                    const cv::Mat& samples) {
    std::vector<float> min_vals(constants::FEATURE_DIMS, std::numeric_limits<float>::max());
    std::vector<float> max_vals(constants::FEATURE_DIMS, std::numeric_limits<float>::lowest());

    std::vector<int> dim_indices(constants::FEATURE_DIMS);
    std::iota(dim_indices.begin(), dim_indices.end(), 0);

    if (!samples.empty()) {
        int numPoints = samples.rows;
        std::vector<int> point_indices(numPoints);
        std::iota(point_indices.begin(), point_indices.end(), 0);
        std::for_each(point_indices.begin(), point_indices.end(), [&](int i) {
            const float* rowPtr = samples.ptr<float>(i);
            std::for_each(dim_indices.begin(), dim_indices.end(), [&](int d) {
                float val = rowPtr[d];
                if (val < min_vals[d])
                    min_vals[d] = val;
                if (val > max_vals[d])
                    max_vals[d] = val;
            });
        });
    } else {
        // Compute from centers if samples matrix is empty
        std::for_each(initialCenters.begin(), initialCenters.end(), [&](const auto& c) {
            std::for_each(dim_indices.begin(), dim_indices.end(), [&](int d) {
                if (c[d] < min_vals[d])
                    min_vals[d] = c[d];
                if (c[d] > max_vals[d])
                    max_vals[d] = c[d];
            });
        });
    }

    float max_range = 0.0f;
    std::for_each(dim_indices.begin(), dim_indices.end(), [&](int d) {
        float range = max_vals[d] - min_vals[d];
        if (range > max_range)
            max_range = range;
    });
    float global_range = max_range + constants::QUANTUM_RANGE_EPSILON;
    m_scaleFactor = (constants::PI_F / 2.0f) / global_range;
}

void QuantumEngine::launchAssignKernelImpl(float* d_samples, int numPoints, float* d_centers, int k, int* d_labels,
                                           int* d_changed, int threadsPerBlock, int blocksPerGrid, size_t sharedSize) {
    quantumAssignKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_samples, numPoints, d_centers, k, d_labels,
                                                                        d_changed, m_scaleFactor);
}

} // namespace kmeans::clustering
