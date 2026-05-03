#include "clustering/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "common/constants.hpp"
#include "common/vector_math.hpp"

namespace kmeans::clustering::metrics {

static float sqDistance(const float* p1, const FeatureVector& p2) {
    return common::VectorMath<constants::FEATURE_DIMS>::sqDistance(p1, p2);
}

BenchmarkResults computeAllMetrics(const cv::Mat& samples,
                                   const std::vector<FeatureVector>& centers, int iterations,
                                   float executionTimeMs) {
    int numPoints = samples.rows;
    int k = static_cast<int>(centers.size());

    if (numPoints == 0 || k == 0) [[unlikely]] {
        return {0.0f, 0.0f, 0.0f, iterations, executionTimeMs};
    }

    // 1. Assign labels and compute WCSS (Inertia) simultaneously
    std::vector<int> labels(numPoints);
    float wcss = 0.0f;
    std::vector<float> intraClusterScatter(k, 0.0f);
    std::vector<int> clusterCounts(k, 0);

    std::vector<int> point_indices(numPoints);
    std::iota(point_indices.begin(), point_indices.end(), 0);
    std::for_each(point_indices.begin(), point_indices.end(), [&](int i) {
        const auto* p = samples.ptr<float>(i);
        float minDistSq = constants::MATH_INF;
        int bestK = 0;
        std::vector<int> k_indices_inner(k);
        std::iota(k_indices_inner.begin(), k_indices_inner.end(), 0);
        std::for_each(k_indices_inner.begin(), k_indices_inner.end(), [&](int j) {
            float d2 = sqDistance(p, centers[j]);
            if (d2 < minDistSq) {
                minDistSq = d2;
                bestK = j;
            }
        });
        labels[i] = bestK;
        wcss += minDistSq;
        intraClusterScatter[bestK] += std::sqrt(minDistSq); // Davies-Bouldin uses linear distance
        clusterCounts[bestK]++;
    });

    // 2. Davies-Bouldin Index
    std::vector<int> k_indices(k);
    std::iota(k_indices.begin(), k_indices.end(), 0);
    std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
        if (clusterCounts[j] > 0) {
            intraClusterScatter[j] /= static_cast<float>(clusterCounts[j]);
        }
    });

    float daviesBouldin = 0.0f;
    std::for_each(k_indices.begin(), k_indices.end(), [&](int i) {
        float maxR = 0.0f;
        std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
            if (i == j) {
                return;
            }
            float dCenter = std::sqrt(common::VectorMath<constants::FEATURE_DIMS>::sqDistance(centers[i].val, centers[j].val));
            if (dCenter > constants::MATH_EPSILON) [[likely]] {
                maxR = std::max((intraClusterScatter[i] + intraClusterScatter[j]) / dCenter, maxR);
            }
        });
        daviesBouldin += maxR;
    });
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

    std::vector<int> subset_indices(subsetSize);
    std::iota(subset_indices.begin(), subset_indices.end(), 0);
    std::for_each(subset_indices.begin(), subset_indices.end(), [&](int idx1) {
        int i = indices[idx1];
        const auto* p = samples.ptr<float>(i);
        int myK = labels[i];

        if (clusterCounts[myK] <= 1) {
            return;
        }

        float a = 0.0f;
        int aCount = 0;
        std::vector<float> bDistSum(k, 0.0f);
        std::vector<int> bCount(k, 0);

        std::for_each(subset_indices.begin(), subset_indices.end(), [&](int idx2) {
            int j = indices[(idx1 + idx2 + 1) % numPoints];
            if (i == j) {
                return;
            }

            const auto* p2 = samples.ptr<float>(j);
            float d = std::sqrt(common::VectorMath<constants::FEATURE_DIMS>::sqDistance(p, p2));

            int otherK = labels[j];
            if (otherK == myK) {
                a += d;
                aCount++;
            } else {
                bDistSum[otherK] += d;
                bCount[otherK]++;
            }
        });

        a = (aCount > 0) ? (a / static_cast<float>(aCount)) : 0.0f;

        float b = constants::MATH_INF;
        std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
            if (j == myK) {
                return;
            }
            if (bCount[j] > 0) {
                b = std::min(bDistSum[j] / static_cast<float>(bCount[j]), b);
            }
        });

        float maxAB = std::max(a, b);
        if (maxAB > constants::MATH_EPSILON && b < constants::MATH_INF) [[likely]] {
            totalSilhouette += (b - a) / maxAB;
            validSilhouettePoints++;
        }
    });

    float avgSilhouette =
        (validSilhouettePoints > 0) ? (totalSilhouette / static_cast<float>(validSilhouettePoints)) : 0.0f;

    return {wcss, daviesBouldin, avgSilhouette, iterations, executionTimeMs};
}

} // namespace kmeans::clustering::metrics
