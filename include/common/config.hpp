/**
 * @file config.hpp
 * @brief Configuration structures for the clustering pipeline.
 */

#pragma once

#include "common/constants.hpp"
#include "common/enums.hpp"

namespace kmeans::common {

/**
 * @struct SegmentationConfig
 * @brief Encapsulates all user-adjustable parameters for the segmentation algorithm.
 *
 * This structure acts as the bridge between the UI control panel and the
 * ClusteringManager. Modifying these values will trigger re-initialization
 * or behavior changes in the underlying engines.
 */
struct SegmentationConfig {
    /** @brief Sampling rate (1 = every pixel, 4 = every 4th pixel, etc). Higher is faster but less precise. */
    int stride = constants::clustering::DEFAULT_STRIDE;

    /** @brief Which strategy to use for selecting initial cluster centers. */
    InitializationType init = InitializationType::KMEANS_PLUSPLUS;

    /** @brief Which mathematical engine to use (Classical Euclidean vs. Quantum Simulation). */
    AlgorithmType algorithm = AlgorithmType::KMEANS_REGULAR;

    /** @brief The number of clusters (segments) to find in the image. */
    int k = 3;

    /** @brief How many frames to wait between re-running the K-Means optimization loop. */
    int learningInterval = constants::clustering::DEFAULT_LEARN_INTERVAL;

    /** @brief The maximum number of iterations allowed per K-Means run. */
    int maxIterations = 20;
};

} // namespace kmeans::common
