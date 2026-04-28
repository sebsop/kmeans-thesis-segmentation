#include "common/utils.hpp"

#include "common/constants.hpp"

namespace kmeans::common {

cv::Vec<float, constants::FEATURE_DIMS> makeFeature(const cv::Vec3f& bgr, float x01, float y01) {
    return {bgr[0] * constants::COLOR_SCALE, bgr[1] * constants::COLOR_SCALE, bgr[2] * constants::COLOR_SCALE,
            x01 * constants::SPATIAL_SCALE, y01 * constants::SPATIAL_SCALE};
}

} // namespace kmeans::common