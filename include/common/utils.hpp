#pragma once

#include <opencv2/core.hpp>

#include "common/constants.hpp"
#include "common/enums.hpp"

namespace kmeans::common {

/**
 * @brief Create a 5D feature vector from BGR color and normalized spatial coordinates.
 *
 * Scaled by the color_scale and spatial_scale arguments respectively.
 *
 * @param bgr The BGR color of the pixel (Vec3f)
 * @param x01 Normalized spatial X coordinate of the pixel in the frame ([0, 1] range)
 * @param y01 Normalized spatial Y coordinate of the pixel in the frame ([0, 1] range)
 *
 * @return cv::Vec<float, constants::FEATURE_DIMS> A 5D feature vector with scaled color and spatial components.
 */
[[nodiscard]] cv::Vec<float, constants::FEATURE_DIMS> makeFeature(const cv::Vec3f& bgr, float x01, float y01);

} // namespace kmeans::common