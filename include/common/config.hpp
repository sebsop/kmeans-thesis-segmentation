#pragma once
#include "common/enums.hpp"

namespace kmeans {
    struct SegmentationConfig {
        AlgorithmType algorithm = AlgorithmType::KMEANS_REGULAR;
        InitializationType init = InitializationType::RANDOM;
        DataStrategy strategy = DataStrategy::RCC_TREES;
        int k = 5;
        int learningInterval = 5;
    };
}