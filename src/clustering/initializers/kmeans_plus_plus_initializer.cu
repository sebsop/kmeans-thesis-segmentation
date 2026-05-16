/**
 * @file kmeans_plus_plus_initializer.cu
 * @brief GPU-accelerated implementation of the K-Means++ algorithm.
 */

#include <cuda_runtime.h>
#include <limits>
#include <numeric>
#include <random>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"
#include "common/constants.hpp"
#include "common/vector_math.hpp"

namespace kmeans::clustering {

/**
 * @brief CUDA kernel to update the minimum distance from each point to the nearest centroid.
 *
 * For every point, this kernel computes the distance to the newly added centroid
 * and updates the global minimum distance if the new one is smaller.
 */
__global__ void compute_min_distances_kernel(const float* __restrict__ samples, int numPoints,
                                             const float* __restrict__ latestCenter, float* __restrict__ distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float distSq = common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(
            &samples[idx * constants::clustering::FEATURE_DIMS], latestCenter);
        if (distSq < distances[idx]) {
            distances[idx] = distSq;
        }
    }
}

/**
 * @brief Implements the K-Means++ seeding algorithm using GPU acceleration.
 *
 * Algorithm Workflow:
 * 1. Select the first centroid uniformly at random.
 * 2. For the remaining K-1 centroids:
 *    a. Compute the squared distance D(x)^2 from each point to its nearest centroid on the GPU.
 *    b. Perform an inclusive scan (prefix sum) of D(x)^2 to create a cumulative distribution.
 *    c. Select a random threshold and use binary search (thrust::upper_bound) to pick the next point.
 *
 * This ensures that points farther away from existing centroids have a much
 * higher probability of being selected, leading to faster convergence.
 */
std::vector<FeatureVector> KMeansPlusPlusInitializer::initialize(const cv::Mat& samples, int k) const {
    CV_Assert(samples.isContinuous());

    std::vector<FeatureVector> centers;
    centers.reserve(k);
    int numPoints = samples.rows;

    thread_local static std::mt19937 gen(constants::clustering::STABLE_RANDOM_SEED);
    std::uniform_int_distribution<> dis(0, numPoints - 1);

    // 1. Pick first center at random
    int firstIdx = dis(gen);
    const auto* firstPtr = samples.ptr<float>(firstIdx);
    {
        FeatureVector first_c;
        std::copy_n(firstPtr, constants::clustering::FEATURE_DIMS, first_c.val);
        centers.push_back(first_c);
    }

    if (k <= 1)
        return centers;

    // 2. Prepare GPU resources using Thrust for RAII and performance
    thrust::device_vector<float> d_samples_vec(numPoints * constants::clustering::FEATURE_DIMS);
    cudaMemcpy(thrust::raw_pointer_cast(d_samples_vec.data()), samples.ptr<float>(),
               numPoints * constants::clustering::FEATURE_DIMS * sizeof(float), cudaMemcpyHostToDevice);

    thrust::device_vector<float> d_latestCenter_vec(constants::clustering::FEATURE_DIMS);
    thrust::device_vector<float> d_distances(numPoints, std::numeric_limits<float>::max());
    thrust::device_vector<float> d_cumulative_distances(numPoints);

    int blockSize = constants::cuda::THREADS_PER_BLOCK;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    // Dedicated stream for initialization
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 3. Iteratively pick K-1 more centers
    for (int i = 1; i < k; ++i) {
        // Upload the most recently picked center to GPU
        const auto& latestCenter = centers.back();
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_latestCenter_vec.data()), &latestCenter[0],
                        constants::clustering::FEATURE_DIMS * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Update D(x)^2 on GPU
        compute_min_distances_kernel<<<gridSize, blockSize, 0, stream>>>(
            thrust::raw_pointer_cast(d_samples_vec.data()), numPoints,
            thrust::raw_pointer_cast(d_latestCenter_vec.data()), thrust::raw_pointer_cast(d_distances.data()));

        cudaStreamSynchronize(stream);

        // Compute cumulative distribution (Prefix Sum)
        thrust::inclusive_scan(thrust::cuda::par.on(stream), d_distances.begin(), d_distances.end(),
                               d_cumulative_distances.begin());

        // Fitness-proportional selection
        float sumDistSq = d_cumulative_distances.back();
        std::uniform_real_distribution<float> fdis(0.0f, sumDistSq);
        float target = fdis(gen);

        // Find selected index using binary search on GPU
        auto iter = thrust::upper_bound(thrust::cuda::par.on(stream), d_cumulative_distances.begin(),
                                        d_cumulative_distances.end(), target);
        int selectedIdx = static_cast<int>(thrust::distance(d_cumulative_distances.begin(), iter));

        if (selectedIdx >= numPoints)
            selectedIdx = numPoints - 1;

        // Add to centers list
        const auto* selPtr = samples.ptr<float>(selectedIdx);
        FeatureVector c;
        std::copy_n(selPtr, constants::clustering::FEATURE_DIMS, c.val);
        centers.push_back(c);
    }

    cudaStreamDestroy(stream);
    return centers;
}

} // namespace kmeans::clustering
