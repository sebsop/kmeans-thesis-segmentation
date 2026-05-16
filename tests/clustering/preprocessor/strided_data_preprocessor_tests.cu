/**
 * @file strided_data_preprocessor_tests.cu
 * @brief Unit tests for the GPU-accelerated strided data preprocessor.
 */

#include <cuda_runtime.h>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/preprocessor/strided_data_preprocessor.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Preprocessor {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying GPU-based data preprocessing.
 *
 * Ensures a valid CUDA context and resets the device between tests to prevent
 * memory leakage or side effects.
 */
class Preprocessor_StridedData : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            FAIL() << "No CUDA-capable GPU detected.";
        }
    }

    void SetUp() override { cudaDeviceReset(); }
};

/**
 * @brief Verifies the basic BGR-to-Feature mapping and normalization logic.
 *
 * Checks that pixel colors are normalized to [0,1] and spatial coordinates
 * are correctly injected.
 */
TEST_F(Preprocessor_StridedData, PixelToFeatureMapping) {
    StridedDataPreprocessor preproc;

    // 4x4 image, all white pixels
    cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(255, 255, 255));

    auto samples = preproc.prepare(frame);

    EXPECT_EQ(samples.rows, 16);
    EXPECT_EQ(samples.cols, constants::clustering::FEATURE_DIMS);

    // Features: B, G, R, X_scaled, Y_scaled
    EXPECT_NEAR(samples.at<float>(0, 0), 1.0f, 1e-5); // B
    EXPECT_NEAR(samples.at<float>(0, 1), 1.0f, 1e-5); // G
    EXPECT_NEAR(samples.at<float>(0, 2), 1.0f, 1e-5); // R
    EXPECT_NEAR(samples.at<float>(0, 3), 0.0f, 1e-5); // X
    EXPECT_NEAR(samples.at<float>(0, 4), 0.0f, 1e-5); // Y
}

/**
 * @brief Validates the striding logic used for point reduction.
 *
 * Ensures that increasing the stride correctly reduces the number of
 * samples processed by the clustering engine.
 */
TEST_F(Preprocessor_StridedData, StrideReducesPointCount) {
    StridedDataPreprocessor preproc;

    // 10x10 image
    cv::Mat frame(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));

    int numPoints = 0;
    // Stride of 2 should result in a 5x5 grid (25 points)
    preproc.prepareDevice(frame, 2, numPoints);

    EXPECT_EQ(numPoints, 25);

    auto samples = preproc.download();
    EXPECT_EQ(samples.rows, 25);
}

/**
 * @brief Verifies that spatial coordinates are correctly scaled into the feature space.
 */
TEST_F(Preprocessor_StridedData, SpatialScalingCorrectness) {
    StridedDataPreprocessor preproc;

    const int W = 100;
    const int H = 100;
    cv::Mat frame(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    auto samples = preproc.prepare(frame);

    // Check pixel at the center (50, 50)
    int idx = 50 * W + 50;
    float x_feature = samples.at<float>(idx, 3);
    float y_feature = samples.at<float>(idx, 4);

    float expected_x = (50.0f / W) * constants::video::SPATIAL_WEIGHT;
    float expected_y = (50.0f / H) * constants::video::SPATIAL_WEIGHT;

    EXPECT_NEAR(x_feature, expected_x, 1e-5);
    EXPECT_NEAR(y_feature, expected_y, 1e-5);
}

/**
 * @brief Verifies that the preprocessor handles dynamic frame resolution changes.
 *
 * Ensures that the internal GPU memory pool correctly reallocates without crashing.
 */
TEST_F(Preprocessor_StridedData, HandlesFrameResizing) {
    StridedDataPreprocessor preproc;

    cv::Mat small(4, 4, CV_8UC3, cv::Scalar(0));
    EXPECT_NO_THROW(preproc.prepare(small));

    cv::Mat large(640, 480, CV_8UC3, cv::Scalar(0));
    EXPECT_NO_THROW(preproc.prepare(large));

    EXPECT_NO_THROW(preproc.prepare(small));
}

/**
 * @brief Verifies the validity and accessibility of the GPU device pointer.
 */
TEST_F(Preprocessor_StridedData, DevicePointerIsValid) {
    StridedDataPreprocessor preproc;
    cv::Mat frame(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));

    int numPoints = 0;
    float* d_ptr = preproc.prepareDevice(frame, 1, numPoints);

    ASSERT_NE(d_ptr, nullptr);

    // Verify GPU-to-CPU data integrity via manual cudaMemcpy
    std::vector<float> h_first_pixel(5);
    cudaMemcpy(h_first_pixel.data(), d_ptr, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_first_pixel[0], 128.0f / 255.0f, 1e-3);
}

/**
 * @brief Verifies that extreme pixel values (0 and 255) are correctly normalized.
 */
TEST_F(Preprocessor_StridedData, NormalizationBounds) {
    StridedDataPreprocessor preproc;
    cv::Mat frame(1, 2, CV_8UC3);
    frame.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 0);       // Pure Black
    frame.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 255, 255); // Pure White

    int numPoints = 0;
    float* d_ptr = preproc.prepareDevice(frame, 1, numPoints);

    std::vector<float> h_data(10); // 2 pixels * 5 features
    cudaMemcpy(h_data.data(), d_ptr, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    // Check Black pixel (should be 0.0)
    EXPECT_NEAR(h_data[0], 0.0f, 1e-4);
    // Check White pixel (should be 1.0)
    EXPECT_NEAR(h_data[5], 1.0f, 1e-4);
}

} // namespace ThesisTests::Clustering::Preprocessor
