#pragma once
#include <memory>
#include "common/config.hpp"
#include "clustering/data_preprocessor.hpp"
#include "clustering/initializer.hpp"
#include "clustering/kmeans_engine.hpp"

namespace kmeans {
namespace clustering {

    class ClusteringFactory {
    public:
        // Factory methods to encapsulate creation logic
        static std::unique_ptr<DataPreprocessor> createDataPreprocessor(const SegmentationConfig& config);
        static std::unique_ptr<Initializer> createInitializer(const SegmentationConfig& config);
        static std::unique_ptr<KMeansEngine> createEngine(const SegmentationConfig& config);
    };

}
}
