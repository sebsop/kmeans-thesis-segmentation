#include "clustering/clustering_factory.hpp"
#include "common/enums.hpp"

#include "clustering/full_data_preprocessor.hpp"
#include "clustering/rcc_data_preprocessor.hpp"
#include "clustering/random_initializer.hpp"
#include "clustering/kmeans_plus_plus_initializer.hpp"
#include "clustering/classical_engine.hpp"

#include <opencv2/core.hpp>

namespace kmeans {
namespace clustering {

    std::unique_ptr<DataPreprocessor> ClusteringFactory::createDataPreprocessor(const SegmentationConfig& config) {
        if (config.strategy == DataStrategy::RCC_TREES) {
            return std::make_unique<RccDataPreprocessor>();
        } else if (config.strategy == DataStrategy::FULL_DATA) {
            return std::make_unique<FullDataPreprocessor>();
        }
        CV_Assert(false && "Unknown DataStrategy configuration.");
        return nullptr;
    }

    std::unique_ptr<Initializer> ClusteringFactory::createInitializer(const SegmentationConfig& config) {
        if (config.init == InitializationType::KMEANS_PLUSPLUS) {
            return std::make_unique<KMeansPlusPlusInitializer>();
        } else if (config.init == InitializationType::RANDOM) {
            return std::make_unique<RandomInitializer>();
        }
        CV_Assert(false && "Unknown InitializationType configuration.");
        return nullptr;
    }

    std::unique_ptr<KMeansEngine> ClusteringFactory::createEngine(const SegmentationConfig& config) {
        // Extension point for Quantum engine
        // if (config.algorithm == AlgorithmType::QUANTUM) return std::make_unique<QuantumEngine>();
        return std::make_unique<ClassicalEngine>();
    }

}
}
