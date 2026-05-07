/**
 * @file utils.hpp
 * @brief General utility functions for CUDA and error handling.
 */

#pragma once

#include <concepts>
#include <source_location>
#include <string_view>

#include <opencv2/core.hpp>

#include "common/constants.hpp"
#include "common/enums.hpp"

namespace kmeans::common {

/**
 * @brief Calculates the grid dimension needed for a CUDA kernel launch.
 *
 * Given the total number of items to process and the number of threads
 * per block, this helper calculates the number of blocks required to
 * cover all items, rounding up if necessary.
 *
 * @tparam T1 Integral type for total items.
 * @tparam T2 Integral type for threads per block.
 * @param totalItems Total number of elements to process.
 * @param threadsPerBlock Number of threads in a single CUDA block.
 * @return The calculated number of blocks (grid dimension).
 */
template <typename T1, typename T2>
    requires std::integral<T1> && std::integral<T2>
[[nodiscard]] constexpr int calculateGridDim(T1 totalItems, T2 threadsPerBlock) noexcept {
    return (static_cast<int>(totalItems) + static_cast<int>(threadsPerBlock) - 1) / static_cast<int>(threadsPerBlock);
}

} // namespace kmeans::common