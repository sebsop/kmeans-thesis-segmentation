/**
 * @file clustering_factory.hpp
 * @brief Factory Pattern for creating clustering components.
 */

#pragma once

#include <memory>

#include "clustering/engines/kmeans_engine.hpp"
#include "clustering/initializers/initializer.hpp"
#include "common/config.hpp"

namespace kmeans::clustering {

/**
 * @class ClusteringFactory
 * @brief Centralized factory for instantiating clustering components.
 *
 * This class implements the Factory Design Pattern to decouple the application
 * logic from specific algorithm implementations. It reads the
 * SegmentationConfig and returns the appropriate concrete implementations of
 * Initializer and KMeansEngine.
 *
 * By using this factory, the rest of the system can work with abstract
 * interfaces (Strategy Pattern), making it easy to add new algorithms
 * (e.g., different quantum simulation methods if we so wish) in the future.
 */
class ClusteringFactory {
  public:
    /**
     * @brief Creates a centroid initializer based on config settings.
     * @param config The global application configuration.
     * @return A unique_ptr to the chosen concrete Initializer (e.g., Random or K-Means++).
     */
    [[nodiscard]] static std::unique_ptr<Initializer> createInitializer(const common::SegmentationConfig& config);

    /**
     * @brief Creates a K-Means engine based on config settings.
     * @param config The global application configuration.
     * @return A unique_ptr to the chosen concrete KMeansEngine (e.g., Classical or Quantum).
     */
    [[nodiscard]] static std::unique_ptr<KMeansEngine> createEngine(const common::SegmentationConfig& config);
};

} // namespace kmeans::clustering
