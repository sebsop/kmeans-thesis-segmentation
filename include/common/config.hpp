#pragma once
#include "common/enums.hpp"

namespace kmeans {
    struct SegmentationConfig {
        Algorithm algorithm = Algorithm::KMEANS_REGULAR;
        InitializationType init = InitializationType::RANDOM;
        DataStrategy strategy = DataStrategy::RCC_TREES;
        int k = 5;
        int learningInterval = 5;
    };
}