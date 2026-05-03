#pragma once

#include <cmath>
#include <concepts>

#include <opencv2/core.hpp>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef _MSC_VER
#define FORCE_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__)
#define FORCE_INLINE [[gnu::always_inline]]
#else
#define FORCE_INLINE inline
#endif

namespace kmeans::common {

template <int Dims> struct VectorMath {
    /**
     * @brief Computes squared Euclidean distance between two vectors.
     */
    template <typename T1, typename T2>
        requires std::floating_point<T1> && std::floating_point<T2>
    FORCE_INLINE static __host__ __device__ inline float sqDistance(const T1* a, const T2* b) {
        float dist = 0.0f;
#ifdef __CUDACC__
#pragma unroll
#endif
        for (int i = 0; i < Dims; ++i) {
            float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
            dist += diff * diff;
        }
        return dist;
    }

    /**
     * @brief Computes squared Euclidean distance between a pointer and a cv::Vec.
     */
    template <typename T1, typename T2>
        requires std::floating_point<T1> && std::floating_point<T2>
    FORCE_INLINE static __host__ __device__ inline float sqDistance(const T1* a, const cv::Vec<T2, Dims>& b) {
        float dist = 0.0f;
#ifdef __CUDACC__
#pragma unroll
#endif
        for (int i = 0; i < Dims; ++i) {
            float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
            dist += diff * diff;
        }
        return dist;
    }

    /**
     * @brief Accumulates source elements into destination.
     */
    template <typename T1, typename T2>
        requires std::floating_point<T1> && std::floating_point<T2>
    FORCE_INLINE static __host__ __device__ inline void accumulate(T1* dest, const T2* src) {
#ifdef __CUDACC__
#pragma unroll
#endif
        for (int i = 0; i < Dims; ++i) {
            dest[i] += static_cast<T1>(src[i]);
        }
    }
};

} // namespace kmeans::common
