#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "common/vector_math.hpp"

namespace ThesisTests::Common {

using namespace kmeans;
using namespace kmeans::common;

class Common_VectorMath : public ::testing::Test {};

// 1. VectorMath: Squared Distance (Static Dims)
TEST_F(Common_VectorMath, SquaredDistanceCorrectness) {
    const int D = 3;
    float p1[D] = {1.0f, 2.0f, 3.0f};
    float p2[D] = {4.0f, 6.0f, 8.0f};
    
    // distSq = (4-1)^2 + (6-2)^2 + (8-3)^2 = 3^2 + 4^2 + 5^2 = 9 + 16 + 25 = 50
    float result = VectorMath<D>::sqDistance(p1, p2);
    EXPECT_NEAR(result, 50.0f, 1e-5);
}

// 2. VectorMath: Accumulation
TEST_F(Common_VectorMath, AccumulateCorrectness) {
    const int D = 3;
    float dest[D] = {1.0f, 1.0f, 1.0f};
    float src[D] = {2.0f, 3.0f, 4.0f};
    
    VectorMath<D>::accumulate(dest, src);
    
    EXPECT_NEAR(dest[0], 3.0f, 1e-5);
    EXPECT_NEAR(dest[1], 4.0f, 1e-5);
    EXPECT_NEAR(dest[2], 5.0f, 1e-5);
}

// 3. VectorMath: cv::Vec compatibility
TEST_F(Common_VectorMath, OpenCVVecDistance) {
    const int D = 3;
    float p1[D] = {0.0f, 0.0f, 0.0f};
    cv::Vec<float, D> p2(3.0f, 4.0f, 0.0f);
    
    float result = VectorMath<D>::sqDistance(p1, p2);
    EXPECT_NEAR(result, 25.0f, 1e-5); // 3^2 + 4^2 = 9 + 16 = 25
}

} // namespace ThesisTests::Common
