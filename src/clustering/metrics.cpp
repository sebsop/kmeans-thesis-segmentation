#include "clustering/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "common/constants.hpp"
#include "common/vector_math.hpp"

namespace kmeans::clustering::metrics {

static float sqDistance(const float* p1, const FeatureVector& p2) {
    return common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(p1, p2);
}

BenchmarkResults computeAllMetrics(const cv::Mat& samples, const std::vector<FeatureVector>& centers, int iterations,
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
        const auto* currentPoint = samples.ptr<float>(i);
        float minDistSq = constants::math::INF;
        int bestK = 0;
        std::vector<int> k_indices_inner(k);
        std::iota(k_indices_inner.begin(), k_indices_inner.end(), 0);
        std::for_each(k_indices_inner.begin(), k_indices_inner.end(), [&](int j) {
            float distanceSq = sqDistance(currentPoint, centers[j]);
            if (distanceSq < minDistSq) {
                minDistSq = distanceSq;
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
        float maxClusterRatio = 0.0f;
        std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
            if (i == j) {
                return;
            }
            float dCenter = std::sqrt(
                common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(centers[i].val, centers[j].val));
            if (dCenter > constants::math::EPSILON) [[likely]] {
                maxClusterRatio =
                    std::max((intraClusterScatter[i] + intraClusterScatter[j]) / dCenter, maxClusterRatio);
            }
        });
        daviesBouldin += maxClusterRatio;
    });
    daviesBouldin = daviesBouldin / static_cast<float>(k);

    // 3. Approximate Silhouette Score
    // O(N^2) on 300k points is impossible. We approximate using a random subset of 2000 points
    // evaluated against another subset of 2000 points. (4 million operations -> instantly fast).
    int subsetSize = std::min(numPoints, constants::metrics::APPROX_SUBSET_SIZE);
    std::vector<int> indices(numPoints);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(constants::clustering::STABLE_RANDOM_SEED); // Fixed seed for stable comparisons between algorithms
    std::shuffle(indices.begin(), indices.end(), gen);

    float totalSilhouette = 0.0f;
    int validSilhouettePoints = 0;

    std::vector<int> subset_indices(subsetSize);
    std::iota(subset_indices.begin(), subset_indices.end(), 0);
    std::for_each(subset_indices.begin(), subset_indices.end(), [&](int idx1) {
        int i = indices[idx1];
        const auto* currentPoint = samples.ptr<float>(i);
        int currentCluster = labels[i];

        if (clusterCounts[currentCluster] <= 1) {
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
            float d = std::sqrt(common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(currentPoint, p2));

            int comparisonCluster = labels[j];
            if (comparisonCluster == currentCluster) {
                a += d;
                aCount++;
            } else {
                bDistSum[comparisonCluster] += d;
                bCount[comparisonCluster]++;
            }
        });

        a = (aCount > 0) ? (a / static_cast<float>(aCount)) : 0.0f;

        float b = constants::math::INF;
        std::for_each(k_indices.begin(), k_indices.end(), [&](int j) {
            if (j == currentCluster) {
                return;
            }
            if (bCount[j] > 0) {
                b = std::min(bDistSum[j] / static_cast<float>(bCount[j]), b);
            }
        });

        float maxAB = std::max(a, b);
        if (maxAB > constants::math::EPSILON && b < constants::math::INF) [[likely]] {
            totalSilhouette += (b - a) / maxAB;
            validSilhouettePoints++;
        }
    });

    float avgSilhouette =
        (validSilhouettePoints > 0) ? (totalSilhouette / static_cast<float>(validSilhouettePoints)) : 0.0f;

    return {wcss, daviesBouldin, avgSilhouette, iterations, executionTimeMs};
}

} // namespace kmeans::clustering::metrics
