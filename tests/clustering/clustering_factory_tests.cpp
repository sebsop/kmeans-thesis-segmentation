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

class Clustering_Factory : public ::testing::Test {};

// 1. Engine Creation (Polymorphism Check)
TEST_F(Clustering_Factory, CreatesClassicalEngine) {
    common::SegmentationConfig config;
    config.algorithm = common::AlgorithmType::KMEANS_REGULAR;

    auto engine = ClusteringFactory::createEngine(config);

    ASSERT_NE(engine, nullptr);
    // Verify it's actually a ClassicalEngine
    EXPECT_NE(dynamic_cast<ClassicalEngine*>(engine.get()), nullptr);
    EXPECT_EQ(dynamic_cast<QuantumEngine*>(engine.get()), nullptr);
}

TEST_F(Clustering_Factory, CreatesQuantumEngine) {
    common::SegmentationConfig config;
    config.algorithm = common::AlgorithmType::KMEANS_QUANTUM;

    auto engine = ClusteringFactory::createEngine(config);

    ASSERT_NE(engine, nullptr);
    // Verify it's actually a QuantumEngine
    EXPECT_NE(dynamic_cast<QuantumEngine*>(engine.get()), nullptr);
    EXPECT_EQ(dynamic_cast<ClassicalEngine*>(engine.get()), nullptr);
}

// 2. Initializer Creation
TEST_F(Clustering_Factory, CreatesKMeansPlusPlusInitializer) {
    common::SegmentationConfig config;
    config.init = common::InitializationType::KMEANS_PLUSPLUS;

    auto init = ClusteringFactory::createInitializer(config);

    ASSERT_NE(init, nullptr);
    EXPECT_NE(dynamic_cast<KMeansPlusPlusInitializer*>(init.get()), nullptr);
}

TEST_F(Clustering_Factory, CreatesRandomInitializer) {
    common::SegmentationConfig config;
    config.init = common::InitializationType::RANDOM;

    auto init = ClusteringFactory::createInitializer(config);

    ASSERT_NE(init, nullptr);
    EXPECT_NE(dynamic_cast<RandomInitializer*>(init.get()), nullptr);
}

// 3. Robustness check for invalid/default config
TEST_F(Clustering_Factory, HandlesDefaultConfig) {
    common::SegmentationConfig config; // Default ctor

    auto engine = ClusteringFactory::createEngine(config);
    auto init = ClusteringFactory::createInitializer(config);

    EXPECT_NE(engine, nullptr);
    EXPECT_NE(init, nullptr);
}

} // namespace ThesisTests::Clustering
