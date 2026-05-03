#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "common/utils.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Common {

using namespace kmeans;
using namespace kmeans::common;

class Common_Utils : public ::testing::Test {};

// 1. Grid Dimension Calculation (CUDA correctness)
TEST_F(Common_Utils, CalculateGridDimCorrectness) {
    EXPECT_EQ(calculateGridDim(100, 32), 4);   // 100/32 = 3.125 -> 4
    EXPECT_EQ(calculateGridDim(64, 32), 2);    // 64/32 = 2.0 -> 2
    EXPECT_EQ(calculateGridDim(0, 32), 0);     // 0 items
    EXPECT_EQ(calculateGridDim(1, 32), 1);     // 1 item
}

// 2. Feature Vector Creation
TEST_F(Common_Utils, MakeFeatureScaling) {
    cv::Vec3f bgr(255.0f, 127.5f, 0.0f);
    float x01 = 0.5f;
    float y01 = 0.8f;
    
    auto feature = makeFeature(bgr, x01, y01);
    
    // B, G, R scaled by 1/255.0
    EXPECT_NEAR(feature[0], 1.0f, 1e-5);
    EXPECT_NEAR(feature[1], 0.5f, 1e-5);
    EXPECT_NEAR(feature[2], 0.0f, 1e-5);
    
    // X, Y scaled by SPATIAL_SCALE
    EXPECT_NEAR(feature[3], 0.5f * constants::video::SPATIAL_SCALE, 1e-5);
    EXPECT_NEAR(feature[4], 0.8f * constants::video::SPATIAL_SCALE, 1e-5);
}

} // namespace ThesisTests::Common
