#include "common/utils.hpp"

#include "common/constants.hpp"

namespace kmeans::common {

FeatureVector makeFeature(const cv::Vec3f& bgr, float x01, float y01) {
    return {bgr[0] * constants::video::COLOR_SCALE, bgr[1] * constants::video::COLOR_SCALE,
            bgr[2] * constants::video::COLOR_SCALE, x01 * constants::video::SPATIAL_SCALE,
            y01 * constants::video::SPATIAL_SCALE};
}

} // namespace kmeans::common