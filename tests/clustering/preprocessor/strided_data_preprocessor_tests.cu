#include <cuda_runtime.h>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/preprocessor/strided_data_preprocessor.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Preprocessor {

using namespace kmeans;
using namespace kmeans::clustering;

class Preprocessor_StridedData : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            FAIL() << "No CUDA-capable GPU detected.";
        }
    }

    void SetUp() override {
        // Ensure a clean slate for each test
        cudaDeviceReset();
    }
};

// 1. Basic Mapping and Normalization
TEST_F(Preprocessor_StridedData, PixelToFeatureMapping) {
    StridedDataPreprocessor preproc;

    // 4x4 image, all white pixels
    cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(255, 255, 255));

    auto samples = preproc.prepare(frame);

    EXPECT_EQ(samples.rows, 16);
    EXPECT_EQ(samples.cols, constants::clustering::FEATURE_DIMS);

    // Check first pixel (0,0)
    // Features: B, G, R, X_scaled, Y_scaled
    EXPECT_NEAR(samples.at<float>(0, 0), 1.0f, 1e-5); // B
    EXPECT_NEAR(samples.at<float>(0, 1), 1.0f, 1e-5); // G
    EXPECT_NEAR(samples.at<float>(0, 2), 1.0f, 1e-5); // R
    EXPECT_NEAR(samples.at<float>(0, 3), 0.0f, 1e-5); // X
    EXPECT_NEAR(samples.at<float>(0, 4), 0.0f, 1e-5); // Y
}

// 2. Striding Logic (Point Reduction)
TEST_F(Preprocessor_StridedData, StrideReducesPointCount) {
    StridedDataPreprocessor preproc;

    // 10x10 image
    cv::Mat frame(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));

    int numPoints = 0;
    // Stride of 2 should pick (0,0), (0,2), (0,4), (0,6), (0,8) -> 5 per row/col
    // Total: 5 * 5 = 25
    preproc.prepareDevice(frame, 2, numPoints);

    EXPECT_EQ(numPoints, 25);

    auto samples = preproc.download();
    EXPECT_EQ(samples.rows, 25);
}

// 3. Spatial Scaling Verification
TEST_F(Preprocessor_StridedData, SpatialScalingCorrectness) {
    StridedDataPreprocessor preproc;

    // 100x100 image
    const int W = 100;
    const int H = 100;
    cv::Mat frame(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    auto samples = preproc.prepare(frame);

    // Check pixel at (50, 50)
    // index = 50 * 100 + 50 = 5050
    int idx = 50 * W + 50;
    float x_feature = samples.at<float>(idx, 3);
    float y_feature = samples.at<float>(idx, 4);

    float expected_x = (50.0f / W) * constants::video::SPATIAL_SCALE;
    float expected_y = (50.0f / H) * constants::video::SPATIAL_SCALE;

    EXPECT_NEAR(x_feature, expected_x, 1e-5);
    EXPECT_NEAR(y_feature, expected_y, 1e-5);
}

// 4. Memory Stability (Resizing)
TEST_F(Preprocessor_StridedData, HandlesFrameResizing) {
    StridedDataPreprocessor preproc;

    // Small frame
    cv::Mat small(4, 4, CV_8UC3, cv::Scalar(0));
    EXPECT_NO_THROW(preproc.prepare(small));

    // Large frame (triggers realloc)
    cv::Mat large(640, 480, CV_8UC3, cv::Scalar(0));
    EXPECT_NO_THROW(preproc.prepare(large));

    // Back to small
    EXPECT_NO_THROW(preproc.prepare(small));
}

// 5. Device Pointer Persistence
TEST_F(Preprocessor_StridedData, DevicePointerIsValid) {
    StridedDataPreprocessor preproc;
    cv::Mat frame(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));

    int numPoints = 0;
    float* d_ptr = preproc.prepareDevice(frame, 1, numPoints);

    ASSERT_NE(d_ptr, nullptr);

    // Verify we can actually copy from it
    std::vector<float> h_first_pixel(5);
    cudaMemcpy(h_first_pixel.data(), d_ptr, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_first_pixel[0], 128.0f / 255.0f, 1e-3);
}

} // namespace ThesisTests::Clustering::Preprocessor
