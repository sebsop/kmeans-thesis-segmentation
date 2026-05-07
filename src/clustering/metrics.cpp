/**
 * @file metrics.cpp
 * @brief Implementation of mathematical quality metrics for clustering evaluation.
 */

#include "clustering/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "common/constants.hpp"
#include "common/vector_math.hpp"

namespace kmeans::clustering::metrics {

/** @brief Helper to compute squared Euclidean distance between raw data and a FeatureVector. */
static float sqDistance(const float* p1, const FeatureVector& p2) {
    return common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(p1, p2);
}

/**
 * @brief Computes a suite of quality metrics for a given clustering result.
 *
 * This function calculates:
 *
 * 1. **WCSS (Within-Cluster Sum of Squares)**:
 *    Also known as 'Inertia'. It measures how tightly grouped the points are
 *    around their centroids. Lower values indicate better compactness.
 *    Formula: Σ ||x - μ_i||^2
 *
 * 2. **Davies-Bouldin Index**:
 *    Calculates the average similarity between each cluster and its most
 *    similar one. Similarity is defined as the ratio of within-cluster
 *    scatter to between-cluster separation. Lower values indicate
 *    better separation.
 *
 * 3. **Silhouette Score**:
 *    Measures how similar an object is to its own cluster (cohesion)
 *    compared to other clusters (separation).
 *    - +1: Point is perfectly clustered.
 *    -  0: Point is on the boundary between clusters.
 *    - -1: Point is likely assigned to the wrong cluster.
 *
 * @note **Monte Carlo Approximation**: Since calculating the true Silhouette Score
 *       requires O(N^2) distance checks (impossible for real-time video), we use
 *       a Monte Carlo approach, sampling a fixed subset of points to estimate
 *       the global quality.
 *
 * @param samples The preprocessed feature matrix.
 * @param centers The final cluster centroids.
 * @param iterations The number of iterations the algorithm took to converge.
 * @param executionTimeMs Time taken in milliseconds.
 * @return BenchmarkResults Aggregated metrics.
 */
BenchmarkResults computeAllMetrics(const cv::Mat& samples, const std::vector<FeatureVector>& centers, int iterations,
                                   float executionTimeMs) {
    const int numPoints = samples.rows;
    const int k = static_cast<int>(centers.size());

    if (numPoints == 0 || k == 0) [[unlikely]] {
        return {.wcss = 0.0f,
                .daviesBouldin = 0.0f,
                .silhouetteScore = 0.0f,
                .iterations = iterations,
                .executionTimeMs = executionTimeMs};
    }

    // -------------------------------------------------------------------------
    // 1. WCSS & Initial Labeling
    // -------------------------------------------------------------------------
    std::vector<int> labels(numPoints);
    float wcss = 0.0f;
    std::vector<float> intraClusterScatterSum(k, 0.0f);
    std::vector<int> clusterCounts(k, 0);

    std::vector<int> pointIndices(numPoints);
    std::iota(pointIndices.begin(), pointIndices.end(), 0);
    std::ranges::for_each(pointIndices, [&](int i) {
        const auto* currentPoint = samples.ptr<float>(i);
        float minDistSq = constants::math::INF;
        int bestClusterIdx = 0;

        for (int j = 0; j < k; ++j) {
            float distanceSq = sqDistance(currentPoint, centers[j]);
            if (distanceSq < minDistSq) {
                minDistSq = distanceSq;
                bestClusterIdx = j;
            }
        }

        labels[i] = bestClusterIdx;
        wcss += minDistSq;                                              // Sum of squared distances (Inertia)
        intraClusterScatterSum[bestClusterIdx] += std::sqrt(minDistSq); // Linear distance for Davies-Bouldin
        clusterCounts[bestClusterIdx]++;
    });

    // -------------------------------------------------------------------------
    // 2. Davies-Bouldin Index Calculation
    // -------------------------------------------------------------------------
    std::vector<int> kIndices(k);
    std::iota(kIndices.begin(), kIndices.end(), 0);

    // Compute 'Scatter' (average distance from each point in a cluster to its centroid)
    std::vector<float> avgIntraClusterScatter(k, 0.0f);
    std::ranges::for_each(kIndices, [&](int j) {
        if (clusterCounts[j] > 0) {
            avgIntraClusterScatter[j] = intraClusterScatterSum[j] / static_cast<float>(clusterCounts[j]);
        }
    });

    float totalDaviesBouldin = 0.0f;
    std::ranges::for_each(kIndices, [&](int i) {
        float worstClusterRatio = 0.0f;
        std::ranges::for_each(kIndices, [&](int j) {
            if (i == j) {
                return;
            }

            // Separation: Distance between cluster centroids
            float centroidDistance = std::sqrt(
                common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(centers[i].val, centers[j].val));

            if (centroidDistance > constants::math::EPSILON) [[likely]] {
                // Similarity Ratio: (Scatter_i + Scatter_j) / Separation_ij
                float similarityRatio = (avgIntraClusterScatter[i] + avgIntraClusterScatter[j]) / centroidDistance;
                worstClusterRatio = std::max(similarityRatio, worstClusterRatio);
            }
        });
        totalDaviesBouldin += worstClusterRatio;
    });
    float finalDaviesBouldin = totalDaviesBouldin / static_cast<float>(k);

    // -------------------------------------------------------------------------
    // 3. Approximate Silhouette Score (Monte Carlo)
    // -------------------------------------------------------------------------
    // At standard 640x480 (VGA) resolution, we have ~307,200 points.
    // A true Silhouette O(N^2) would require ~94 billion distance checks per frame.
    // To maintain real-time performance, we use a Monte Carlo approach:
    // sampling a random representative subset to estimate the global score.
    int subsetSize = std::min(numPoints, constants::metrics::APPROX_SUBSET_SIZE);
    std::vector<int> allIndices(numPoints);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::mt19937 gen(constants::clustering::STABLE_RANDOM_SEED);
    std::shuffle(allIndices.begin(), allIndices.end(), gen);

    float totalSilhouetteScore = 0.0f;
    int validSilhouetteCount = 0;

    std::vector<int> subsetIndices(subsetSize);
    std::iota(subsetIndices.begin(), subsetIndices.end(), 0);

    std::ranges::for_each(subsetIndices, [&](int idx1) {
        int pointIdx = allIndices[idx1];
        const auto* currentPoint = samples.ptr<float>(pointIdx);
        int currentCluster = labels[pointIdx];

        if (clusterCounts[currentCluster] <= 1) {
            return;
        }

        float intraClusterDistSum = 0.0f; // Cohesion sum
        int intraClusterPointsFound = 0;
        std::vector<float> interClusterDistSum(k, 0.0f); // Separation sums
        std::vector<int> interClusterPointCount(k, 0);

        std::ranges::for_each(subsetIndices, [&](int idx2) {
            int neighborIdx = allIndices[(idx1 + idx2 + 1) % numPoints];
            if (pointIdx == neighborIdx) {
                return;
            }

            const auto* neighborPoint = samples.ptr<float>(neighborIdx);
            float pairDistance = std::sqrt(
                common::VectorMath<constants::clustering::FEATURE_DIMS>::sqDistance(currentPoint, neighborPoint));

            int neighborClusterIdx = labels[neighborIdx];
            if (neighborClusterIdx == currentCluster) {
                intraClusterDistSum += pairDistance;
                intraClusterPointsFound++;
            } else {
                interClusterDistSum[neighborClusterIdx] += pairDistance;
                interClusterPointCount[neighborClusterIdx]++;
            }
        });

        // 'avgIntraClusterDist' (Textbook 'a'): mean distance to all other points in the same cluster
        float avgIntraClusterDist =
            (intraClusterPointsFound > 0) ? (intraClusterDistSum / static_cast<float>(intraClusterPointsFound)) : 0.0f;

        // 'minAvgInterClusterDist' (Textbook 'b'): mean distance to the nearest cluster that the point is NOT a part of
        float minAvgInterClusterDist = constants::math::INF;
        std::ranges::for_each(kIndices, [&](int j) {
            if (j == currentCluster) {
                return;
            }
            if (interClusterPointCount[j] > 0) {
                float avgDistToOtherCluster = interClusterDistSum[j] / static_cast<float>(interClusterPointCount[j]);
                minAvgInterClusterDist = std::min(avgDistToOtherCluster, minAvgInterClusterDist);
            }
        });

        // Silhouette coefficient for this point: s(i) = (b - a) / max(a, b)
        float maxCohesionSeparation = std::max(avgIntraClusterDist, minAvgInterClusterDist);
        if (maxCohesionSeparation > constants::math::EPSILON && minAvgInterClusterDist < constants::math::INF)
            [[likely]] {
            totalSilhouetteScore += (minAvgInterClusterDist - avgIntraClusterDist) / maxCohesionSeparation;
            validSilhouetteCount++;
        }
    });

    float finalSilhouette =
        (validSilhouetteCount > 0) ? (totalSilhouetteScore / static_cast<float>(validSilhouetteCount)) : 0.0f;

    return {.wcss = wcss,
            .daviesBouldin = finalDaviesBouldin,
            .silhouetteScore = finalSilhouette,
            .iterations = iterations,
            .executionTimeMs = executionTimeMs};
}

} // namespace kmeans::clustering::metrics
