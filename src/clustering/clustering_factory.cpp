#include "clustering/clustering_factory.hpp"

#include <stdexcept>

#include "clustering/engines/classical_engine.hpp"
#include "clustering/engines/quantum_engine.hpp"
#include "clustering/initializers/kmeans_plus_plus_initializer.hpp"
#include "clustering/initializers/random_initializer.hpp"
#include "clustering/preprocessors/full_data_preprocessor.hpp"
#include "clustering/preprocessors/rcc_data_preprocessor.hpp"

namespace kmeans::clustering {

std::unique_ptr<DataPreprocessor> ClusteringFactory::createDataPreprocessor(const common::SegmentationConfig& config) {
    if (config.strategy == common::DataStrategy::FULL_DATA) {
        return std::make_unique<FullDataPreprocessor>();
    }

    return std::make_unique<RccDataPreprocessor>();
}

std::unique_ptr<Initializer> ClusteringFactory::createInitializer(const common::SegmentationConfig& config) {
    if (config.init == common::InitializationType::RANDOM) {
        return std::make_unique<RandomInitializer>();
    }

    return std::make_unique<KMeansPlusPlusInitializer>();
}

std::unique_ptr<KMeansEngine> ClusteringFactory::createEngine(const common::SegmentationConfig& config) {
    if (config.algorithm == common::AlgorithmType::KMEANS_QUANTUM) {
        return std::make_unique<QuantumEngine>();
    }

    return std::make_unique<ClassicalEngine>();
}

} // namespace kmeans::clustering
