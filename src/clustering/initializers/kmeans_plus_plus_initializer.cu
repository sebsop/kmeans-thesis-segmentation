#include <cuda_runtime.h>
#include <limits>
#include <random>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"
#include "common/constants.hpp"
#include "common/vector_math.hpp"

namespace kmeans::clustering {

__global__ void compute_min_distances_kernel(const float* __restrict__ samples, int numPoints,
                                             const float* __restrict__ latestCenter, float* __restrict__ distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float distSq = common::VectorMath<constants::FEATURE_DIMS>::sqDistance(&samples[idx * constants::FEATURE_DIMS], latestCenter);
        if (distSq < distances[idx]) {
            distances[idx] = distSq;
        }
    }
}

std::vector<cv::Vec<float, constants::FEATURE_DIMS>> KMeansPlusPlusInitializer::initialize(const cv::Mat& samples,
                                                                                           int k) const {
    CV_Assert(samples.isContinuous() && "Samples matrix must be continuous for CUDA transfer");

    std::vector<cv::Vec<float, constants::FEATURE_DIMS>> centers;
    centers.reserve(k);
    int numPoints = samples.rows;

    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    // First center random
    int firstIdx = dis(gen);
    const auto* firstPtr = samples.ptr<float>(firstIdx);
    {
        cv::Vec<float, constants::FEATURE_DIMS> first_c;
        for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
            first_c[d] = firstPtr[d];
        }
        centers.push_back(first_c);
    }

    if (k <= 1) {
        return centers;
    }

    // Use thrust::device_vector for automatic RAII
    thrust::device_vector<float> d_samples_vec(numPoints * constants::FEATURE_DIMS);
    cudaMemcpy(thrust::raw_pointer_cast(d_samples_vec.data()), samples.ptr<float>(),
               numPoints * constants::FEATURE_DIMS * sizeof(float), cudaMemcpyHostToDevice);

    thrust::device_vector<float> d_latestCenter_vec(constants::FEATURE_DIMS);

    // Distances initialized to MAX_FLOAT
    thrust::device_vector<float> d_distances(numPoints, std::numeric_limits<float>::max());
    thrust::device_vector<float> d_cumulative_distances(numPoints);

    int blockSize = constants::CUDA_THREADS_PER_BLOCK;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    // Dedicated stream so we only synchronize this work, not the entire GPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 1; i < k; ++i) {
        // Copy latest center to device (only FEATURE_DIMS floats)
        const auto& latestCenter = centers.back();
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_latestCenter_vec.data()), &latestCenter[0],
                        constants::FEATURE_DIMS * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Update minimum distances on GPU
        compute_min_distances_kernel<<<gridSize, blockSize, 0, stream>>>(
            thrust::raw_pointer_cast(d_samples_vec.data()), numPoints,
            thrust::raw_pointer_cast(d_latestCenter_vec.data()), thrust::raw_pointer_cast(d_distances.data()));

        // Wait only for this stream's work
        cudaStreamSynchronize(stream);

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

        // Fetch selected center directly from CPU memory
        const auto* selPtr = samples.ptr<float>(selectedIdx);
        {
            cv::Vec<float, constants::FEATURE_DIMS> c;
            for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
                c[d] = selPtr[d];
            }
            centers.push_back(c);
        }
    }

    cudaStreamDestroy(stream);

    return centers;
}

} // namespace kmeans::clustering
