#include "clustering/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "common/constants.hpp"

namespace kmeans::clustering::metrics {

static float sqDistance(const float* p1, const cv::Vec<float, constants::FEATURE_DIMS>& p2) {
    float d = 0;
    for (int i = 0; i < constants::FEATURE_DIMS; ++i) {
        float diff = p1[i] - p2[i];
        d += diff * diff;
    }
    return d;
}

BenchmarkResults computeAllMetrics(const cv::Mat& samples,
                                   const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& centers, int iterations,
                                   float executionTimeMs) {
    int numPoints = samples.rows;
    int k = static_cast<int>(centers.size());

    if (numPoints == 0 || k == 0) {
        return {0.0f, 0.0f, 0.0f, iterations, executionTimeMs};
    }

    // 1. Assign labels and compute WCSS (Inertia) simultaneously
    std::vector<int> labels(numPoints);
    float wcss = 0.0f;
    std::vector<float> intraClusterScatter(k, 0.0f);
    std::vector<int> clusterCounts(k, 0);

    for (int i = 0; i < numPoints; ++i) {
        const auto* p = samples.ptr<float>(i);
        float minDistSq = constants::MATH_INF;
        int bestK = 0;
        for (int j = 0; j < k; ++j) {
            float d2 = sqDistance(p, centers[j]);
            if (d2 < minDistSq) {
                minDistSq = d2;
                bestK = j;
            }
        }
        labels[i] = bestK;
        wcss += minDistSq;
        intraClusterScatter[bestK] += std::sqrt(minDistSq); // Davies-Bouldin uses linear distance
        clusterCounts[bestK]++;
    }

    // 2. Davies-Bouldin Index
    for (int j = 0; j < k; ++j) {
        if (clusterCounts[j] > 0) {
            intraClusterScatter[j] /= static_cast<float>(clusterCounts[j]);
        }
    }

    float daviesBouldin = 0.0f;
    for (int i = 0; i < k; ++i) {
        float maxR = 0.0f;
        for (int j = 0; j < k; ++j) {
            if (i == j) {
                continue;
            }
            float dCenter = 0.0f;
            for (int d = 0; d < constants::FEATURE_DIMS; ++d) {
                float diff = centers[i][d] - centers[j][d];
                dCenter += diff * diff;
            }
            dCenter = std::sqrt(dCenter);
            if (dCenter > constants::MATH_EPSILON) {
                maxR = std::max((intraClusterScatter[i] + intraClusterScatter[j]) / dCenter, maxR);
            }
        }
        daviesBouldin += maxR;
    }
    daviesBouldin = daviesBouldin / static_cast<float>(k);

    // 3. Approximate Silhouette Score
    // O(N^2) on 300k points is impossible. We approximate using a random subset of 2000 points
    // evaluated against another subset of 2000 points. (4 million operations -> instantly fast).
    int subsetSize = std::min(numPoints, constants::METRIC_APPROX_SUBSET_SIZE);
    std::vector<int> indices(numPoints);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(constants::STABLE_RANDOM_SEED); // Fixed seed for stable comparisons between algorithms
    std::shuffle(indices.begin(), indices.end(), gen);

    float totalSilhouette = 0.0f;
    int validSilhouettePoints = 0;

    for (int idx1 = 0; idx1 < subsetSize; ++idx1) {
        int i = indices[idx1];
        const auto* p = samples.ptr<float>(i);
        int myK = labels[i];

        // Skip if this cluster only has 1 point (silhouette is undefined/0)
        if (clusterCounts[myK] <= 1) {
            continue;
        }

        float a = 0.0f;
        int aCount = 0;
        std::vector<float> bDistSum(k, 0.0f);
        std::vector<int> bCount(k, 0);

        // Compare against the next `subsetSize` points (or wrap around)
        for (int idx2 = 0; idx2 < subsetSize; ++idx2) {
            int j = indices[(idx1 + idx2 + 1) % numPoints];
            if (i == j) {
                continue;
            }

            const auto* p2 = samples.ptr<float>(j);
            float d = 0.0f;
            for (int dim = 0; dim < constants::FEATURE_DIMS; ++dim) {
                float diff = p[dim] - p2[dim];
                d += diff * diff;
            }
            d = std::sqrt(d);

            int otherK = labels[j];
            if (otherK == myK) {
                a += d;
                aCount++;
            } else {
                bDistSum[otherK] += d;
                bCount[otherK]++;
            }
        }

        a = (aCount > 0) ? (a / static_cast<float>(aCount)) : 0.0f;

        float b = constants::MATH_INF;
        for (int j = 0; j < k; ++j) {
            if (j == myK) {
                continue;
            }
            if (bCount[j] > 0) {
                b = std::min(bDistSum[j] / static_cast<float>(bCount[j]), b);
            }
        }

        float maxAB = std::max(a, b);
        if (maxAB > constants::MATH_EPSILON && b < constants::MATH_INF) {
            totalSilhouette += (b - a) / maxAB;
            validSilhouettePoints++;
        }
    }

    float avgSilhouette =
        (validSilhouettePoints > 0) ? (totalSilhouette / static_cast<float>(validSilhouettePoints)) : 0.0f;

    return {wcss, daviesBouldin, avgSilhouette, iterations, executionTimeMs};
}

} // namespace kmeans::clustering::metrics
