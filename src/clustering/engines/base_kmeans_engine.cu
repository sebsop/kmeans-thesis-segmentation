#include "clustering/engines/base_kmeans_engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    } while (0)

namespace kmeans::clustering {

BaseKMeansEngine::~BaseKMeansEngine() {
    if (d_samples)
        cudaFree(d_samples);
    if (d_centers)
        cudaFree(d_centers);
    if (d_labels)
        cudaFree(d_labels);
    if (d_newSums)
        cudaFree(d_newSums);
    if (d_counts)
        cudaFree(d_counts);
    if (d_changed)
        cudaFree(d_changed);
}

void BaseKMeansEngine::ensureBuffers(int numPoints, int k) {
    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        if (d_samples)
            cudaFree(d_samples);
        if (d_centers)
            cudaFree(d_centers);
        if (d_labels)
            cudaFree(d_labels);
        if (d_newSums)
            cudaFree(d_newSums);
        if (d_counts)
            cudaFree(d_counts);
        if (d_changed)
            cudaFree(d_changed);

        m_maxPoints = std::max(m_maxPoints, static_cast<size_t>(numPoints));
        m_maxK = std::max(m_maxK, k);

        CUDA_CHECK(cudaMalloc(&d_samples, m_maxPoints * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centers, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_newSums, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts, m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    }
}

std::vector<cv::Vec<float, 5>> BaseKMeansEngine::run(const cv::Mat& samples,
                                                     const std::vector<cv::Vec<float, 5>>& initialCenters, int k,
                                                     int maxIterations) {
    int numPoints = samples.rows;
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    ensureBuffers(numPoints, k);
    CUDA_CHECK(cudaMemcpy(d_samples, samples.ptr<float>(0), numPoints * 5 * sizeof(float), cudaMemcpyHostToDevice));
    
    preRunSetup(initialCenters, samples);
    
    return runInternal(d_samples, numPoints, initialCenters, k, maxIterations);
}

std::vector<cv::Vec<float, 5>> BaseKMeansEngine::runOnDevice(float* d_samples_ext, int numPoints,
                                                             const std::vector<cv::Vec<float, 5>>& initialCenters, int k,
                                                             int maxIterations) {
    if (numPoints == 0 || k <= 0)
        return initialCenters;

    if (static_cast<size_t>(numPoints) > m_maxPoints || k > m_maxK) {
        if (d_centers)
            cudaFree(d_centers);
        if (d_labels)
            cudaFree(d_labels);
        if (d_newSums)
            cudaFree(d_newSums);
        if (d_counts)
            cudaFree(d_counts);
        if (d_changed)
            cudaFree(d_changed);

        m_maxPoints = std::max(m_maxPoints, static_cast<size_t>(numPoints));
        m_maxK = std::max(m_maxK, k);

        CUDA_CHECK(cudaMalloc(&d_centers, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, m_maxPoints * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_newSums, m_maxK * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts, m_maxK * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    }

    preRunSetup(initialCenters, cv::Mat());

    return runInternal(d_samples_ext, numPoints, initialCenters, k, maxIterations);
}

std::vector<cv::Vec<float, 5>> BaseKMeansEngine::runInternal(float* d_samp, int numPoints,
                                                             const std::vector<cv::Vec<float, 5>>& initialCenters, int k,
                                                             int maxIterations) {
    size_t centersSize = static_cast<size_t>(k) * 5 * sizeof(float);

    std::vector<float> h_centers(k * 5);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < 5; ++d) {
            h_centers[i * 5 + d] = initialCenters[i][d];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_labels, 0xFF, numPoints * sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedAssignSize = static_cast<size_t>(k) * 5 * sizeof(float);
    size_t sharedUpdateSize = static_cast<size_t>(k) * 5 * sizeof(float) + static_cast<size_t>(k) * sizeof(int);

    std::vector<float> h_newSums(k * 5);
    std::vector<int> h_counts(k);

    int iter = 0;
    for (; iter < maxIterations; ++iter) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        launchAssignKernel(d_samp, numPoints, d_centers, k, d_labels, d_changed, threadsPerBlock, blocksPerGrid, sharedAssignSize);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed == 0) {
            break; // converged
        }

        CUDA_CHECK(cudaMemset(d_newSums, 0, centersSize));
        CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(int)));

        launchUpdateKernel(d_samp, numPoints, k, d_labels, d_newSums, d_counts, threadsPerBlock, blocksPerGrid, sharedUpdateSize);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_newSums.data(), d_newSums, centersSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));

        // CPU center averaging
        for (int j = 0; j < k; ++j) {
            if (h_counts[j] > 0) {
                for (int d = 0; d < 5; ++d) {
                    h_centers[j * 5 + d] = h_newSums[j * 5 + d] / static_cast<float>(h_counts[j]);
                }
            } else if (numPoints > 0) {
                int randomIdx = rand() % numPoints;
                CUDA_CHECK(
                    cudaMemcpy(&h_centers[j * 5], &d_samp[randomIdx * 5], 5 * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
        CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), centersSize, cudaMemcpyHostToDevice));
    }

    m_lastIterations = iter + 1;

    std::vector<cv::Vec<float, 5>> finalCenters(k);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < 5; ++d) {
            finalCenters[i][d] = h_centers[i * 5 + d];
        }
    }
    return finalCenters;
}

} // namespace kmeans::clustering
