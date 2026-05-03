#pragma once
#include "common/constants.hpp"
#include "common/enums.hpp"

namespace kmeans::common {

struct SegmentationConfig {
    int stride = constants::clustering::DEFAULT_STRIDE;
    InitializationType init = InitializationType::KMEANS_PLUSPLUS;
    AlgorithmType algorithm = AlgorithmType::KMEANS_REGULAR;

    int k = 3;
    int learningInterval = constants::clustering::DEFAULT_LEARN_INTERVAL;
    int maxIterations = 20;
};

} // namespace kmeans::common
