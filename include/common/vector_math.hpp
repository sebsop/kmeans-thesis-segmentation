/**
 * @file vector_math.hpp
 * @brief High-performance vector math primitives for CPU and GPU.
 */

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

/**
 * @struct VectorMath
 * @brief Static utility template for dimension-agnostic vector operations.
 *
 * This class provides highly optimized mathematical operations that work
 * seamlessly on both the CPU (Host) and GPU (Device). By using templates
 * and FORCE_INLINE, it allows the compiler to generate the most efficient
 * code for the specific dimensions used in clustering (typically 5D).
 *
 * @tparam Dims The dimensionality of the vectors.
 */
template <int Dims> struct VectorMath {
    /**
     * @brief Computes squared Euclidean distance between two raw buffers.
     *
     * Uses squared distance to avoid the computationally expensive square
     * root operation, which is sufficient for comparison in K-Means.
     *
     * @param a Pointer to the first vector.
     * @param b Pointer to the second vector.
     * @return The squared L2 distance.
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
     *
     * Overload for interoperability with OpenCV vector types.
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
     * @brief Accumulates source elements into a destination buffer.
     *
     * Used primarily in the Update step to sum up features for averaging.
     *
     * @param dest The buffer to add values to.
     * @param src The buffer to read values from.
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
