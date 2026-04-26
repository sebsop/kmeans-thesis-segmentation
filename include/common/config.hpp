#pragma once
#include "common/enums.hpp"

namespace kmeans::common {

struct SegmentationConfig {
    int stride = 4;
    InitializationType init = InitializationType::KMEANS_PLUSPLUS;
    AlgorithmType algorithm = AlgorithmType::KMEANS_REGULAR;

    int k = 3;
    int learningInterval = 15;
};

} // namespace kmeans::common
