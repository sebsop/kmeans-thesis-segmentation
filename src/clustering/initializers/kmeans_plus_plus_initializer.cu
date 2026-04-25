#include <cuda_runtime.h>
#include <limits>
#include <random>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"

namespace kmeans::clustering {

__global__ void compute_min_distances_kernel(const float* __restrict__ samples, int numPoints,
                                             const float* __restrict__ latestCenter, float* __restrict__ distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float distSq = 0.0f;
        for (int d = 0; d < 5; ++d) {
            float diff = samples[idx * 5 + d] - latestCenter[d];
            distSq += diff * diff;
        }
        if (distSq < distances[idx]) {
            distances[idx] = distSq;
        }
    }
}

std::vector<cv::Vec<float, 5>> KMeansPlusPlusInitializer::initialize(const cv::Mat& samples, int k) const {
    CV_Assert(samples.isContinuous() && "Samples matrix must be continuous for CUDA transfer");

    std::vector<cv::Vec<float, 5>> centers;
    centers.reserve(k);
    int numPoints = samples.rows;

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    // First center random
    int firstIdx = dis(gen);
    const auto* firstPtr = samples.ptr<float>(firstIdx);
    centers.emplace_back(firstPtr[0], firstPtr[1], firstPtr[2], firstPtr[3], firstPtr[4]);

    if (k <= 1) {
        return centers;
    }

    // Allocate and copy data to device
    float* d_samples = nullptr;
    cudaMalloc(&d_samples, numPoints * 5 * sizeof(float));
    cudaMemcpy(d_samples, samples.ptr<float>(), numPoints * 5 * sizeof(float), cudaMemcpyHostToDevice);

    float* d_latestCenter = nullptr;
    cudaMalloc(&d_latestCenter, 5 * sizeof(float));

    // Distances initialized to MAX_FLOAT
    thrust::device_vector<float> d_distances(numPoints, std::numeric_limits<float>::max());
    thrust::device_vector<float> d_cumulative_distances(numPoints);

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    for (int i = 1; i < k; ++i) {
        // Copy latest center to device
        const auto& latestCenter = centers.back();
        cudaMemcpy(d_latestCenter, &latestCenter[0], 5 * sizeof(float), cudaMemcpyHostToDevice);

        // Update minimum distances
        compute_min_distances_kernel<<<gridSize, blockSize>>>(d_samples, numPoints, d_latestCenter,
                                                              thrust::raw_pointer_cast(d_distances.data()));
        cudaDeviceSynchronize();

        // Inclusive scan for cumulative probabilities
        thrust::inclusive_scan(d_distances.begin(), d_distances.end(), d_cumulative_distances.begin());

        // Get total sum
        float sumDistSq = d_cumulative_distances.back();

        // Pick target
        std::uniform_real_distribution<float> fdis(0.0f, sumDistSq);
        float target = fdis(gen);

        // Upper bound to find the selected index
        auto iter = thrust::upper_bound(d_cumulative_distances.begin(), d_cumulative_distances.end(), target);
        int selectedIdx = thrust::distance(d_cumulative_distances.begin(), iter);

        // Ensure index is valid
        if (selectedIdx >= numPoints) {
            selectedIdx = numPoints - 1;
        }

        // Fetch selected center directly from CPU memory since we have it
        const auto* selPtr = samples.ptr<float>(selectedIdx);
        centers.emplace_back(selPtr[0], selPtr[1], selPtr[2], selPtr[3], selPtr[4]);
    }

    cudaFree(d_samples);
    cudaFree(d_latestCenter);

    return centers;
}

} // namespace kmeans::clustering
