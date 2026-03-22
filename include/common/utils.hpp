#pragma once
#include <opencv2/core.hpp>

#include "common/enums.hpp"

namespace kmeans {
// Create a 5D feature vector from BGR color and normalized spatial coordinates,
// scaled by the color_scale and spatial_scale arguments respectively
//
// Args:
//   bgr: the BGR color of the pixel (Vec3f)
//   x01, y01: normalized spatial coordinates of the pixel in the frame ([0, 1]
//   range) color_scale: scaling factor for the color dimensions spatial_scale:
//   scaling factor for the spatial dimensions
//
// Returns:
//   A 5D feature vector (Vec<float, 5>) with scaled color and spatial
//   components
cv::Vec<float, 5> makeFeature(const cv::Vec3f& bgr, float x01, float y01);
}  // namespace kmeans