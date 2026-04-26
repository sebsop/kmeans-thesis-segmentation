#pragma once

#include <memory>

#include "clustering/engines/kmeans_engine.hpp"
#include "clustering/initializers/initializer.hpp"
#include "common/config.hpp"

namespace kmeans::clustering {

/**
 * @brief Factory Design Pattern implementation for creating clustering strategies.
 *
 * Encapsulates the instantiation logic of various algorithms, preprocessors,
 * and initializers to promote loose coupling. The returned objects can be
 * used polymorphically through their base interfaces (Strategy Pattern).
 */
class ClusteringFactory {
  public:
    /** @brief Creates the appropriate Initializer strategy based on configuration. */
    [[nodiscard]] static std::unique_ptr<Initializer> createInitializer(const common::SegmentationConfig& config);

    /** @brief Creates the appropriate KMeansEngine strategy based on configuration. */
    [[nodiscard]] static std::unique_ptr<KMeansEngine> createEngine(const common::SegmentationConfig& config);
};

} // namespace kmeans::clustering
