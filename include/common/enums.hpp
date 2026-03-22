#pragma once

namespace kmeans
{
    enum class AlgorithmType : int
    {
        KMEANS_REGULAR = 0,
        KMEANS_QUANTUM = 1
    };

    enum class InitializationType : int
    {
        RANDOM = 0,
        KMEANS_PLUSPLUS = 1
    };

    enum class DataStrategy :int
    {
        FULL_DATA = 0,
		RCC_TREES = 1
    };
}