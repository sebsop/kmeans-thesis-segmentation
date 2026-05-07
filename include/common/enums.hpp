/**
 * @file enums.hpp
 * @brief Enumerations for strategy and algorithm selection.
 */

#pragma once

namespace kmeans::common {

/**
 * @enum InitializationType
 * @brief Defines the available methods for picking starting centroids.
 */
enum class InitializationType {
    RANDOM,         ///< Simple random selection of points from the image.
    KMEANS_PLUSPLUS ///< Smart probabilistic initialization to improve stability.
};

/**
 * @enum AlgorithmType
 * @brief Defines the available computational backends for the K-Means loop.
 */
enum class AlgorithmType {
    KMEANS_REGULAR, ///< Standard CPU/GPU Euclidean distance implementation.
    KMEANS_QUANTUM  ///< Simulated quantum distance estimation.
};

} // namespace kmeans::common
