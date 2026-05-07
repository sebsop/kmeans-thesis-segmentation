/**
 * @file clustering_factory_tests.cpp
 * @brief Unit tests for the ClusteringFactory and its polymorphic object creation.
 */

#include <memory>
#include <typeinfo>

#include <gtest/gtest.h>

#include "clustering/clustering_factory.hpp"
#include "clustering/engines/classical_engine.hpp"
#include "clustering/engines/quantum_engine.hpp"
#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"
#include "clustering/initializers/random_initializer.hpp"
#include "common/config.hpp"
#include "common/enums.hpp"

namespace ThesisTests::Clustering {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying the Factory Pattern implementation.
 */
class Clustering_Factory : public ::testing::Test {};

/**
 * @brief Verifies that the factory correctly instantiates a ClassicalEngine.
 *
 * Uses dynamic_cast to prove the returned interface points to the expected
 * concrete implementation.
 */
TEST_F(Clustering_Factory, CreatesClassicalEngine) {
    common::SegmentationConfig config;
    config.algorithm = common::AlgorithmType::KMEANS_REGULAR;

    auto engine = ClusteringFactory::createEngine(config);

    ASSERT_NE(engine, nullptr);
    EXPECT_NE(dynamic_cast<ClassicalEngine*>(engine.get()), nullptr);
    EXPECT_EQ(dynamic_cast<QuantumEngine*>(engine.get()), nullptr);
}

/**
 * @brief Verifies that the factory correctly instantiates a QuantumEngine.
 */
TEST_F(Clustering_Factory, CreatesQuantumEngine) {
    common::SegmentationConfig config;
    config.algorithm = common::AlgorithmType::KMEANS_QUANTUM;

    auto engine = ClusteringFactory::createEngine(config);

    ASSERT_NE(engine, nullptr);
    EXPECT_NE(dynamic_cast<QuantumEngine*>(engine.get()), nullptr);
    EXPECT_EQ(dynamic_cast<ClassicalEngine*>(engine.get()), nullptr);
}

/**
 * @brief Verifies that the factory correctly maps enums to the KMeans++ Initializer.
 */
TEST_F(Clustering_Factory, CreatesKMeansPlusPlusInitializer) {
    common::SegmentationConfig config;
    config.init = common::InitializationType::KMEANS_PLUSPLUS;

    auto init = ClusteringFactory::createInitializer(config);

    ASSERT_NE(init, nullptr);
    EXPECT_NE(dynamic_cast<KMeansPlusPlusInitializer*>(init.get()), nullptr);
}

/**
 * @brief Verifies that the factory correctly maps enums to the Random Initializer.
 */
TEST_F(Clustering_Factory, CreatesRandomInitializer) {
    common::SegmentationConfig config;
    config.init = common::InitializationType::RANDOM;

    auto init = ClusteringFactory::createInitializer(config);

    ASSERT_NE(init, nullptr);
    EXPECT_NE(dynamic_cast<RandomInitializer*>(init.get()), nullptr);
}

/**
 * @brief Ensures the factory provides safe defaults when given an uninitialized configuration.
 */
TEST_F(Clustering_Factory, HandlesDefaultConfig) {
    common::SegmentationConfig config; // Default constructor

    auto engine = ClusteringFactory::createEngine(config);
    auto init = ClusteringFactory::createInitializer(config);

    EXPECT_NE(engine, nullptr);
    EXPECT_NE(init, nullptr);
}

} // namespace ThesisTests::Clustering
