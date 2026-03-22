#pragma once
#include "enums.hpp"

namespace kmeans {
    struct SegmentationConfig {
        Algorithm algorithm = Algorithm::KMEANS_REGULAR;
        Initialization init = Initialization::RANDOM;
        DataStrategy strategy = DataStrategy::RCC_TREES;
        int k = 5;
        int learningInterval = 5;
    };
}