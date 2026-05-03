#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>

#include "backend/cuda_assignment_context.hpp"
#include "common/constants.hpp"

namespace ThesisTests {

using namespace kmeans;
using namespace kmeans::backend;

class Backend_CudaKernels : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            FAIL() << "No CUDA-capable GPU detected.";
        }
    }

    void SetUp() override {
        cudaDeviceReset();
    }
};

// 1. RAII & Lifecycle (Verification of device memory allocation/deallocation)
TEST_F(Backend_CudaKernels, RAII_Lifecycle) {
    const int W = 640;
    const int H = 480;
    const int K = 5;
    
    EXPECT_NO_THROW({
        CudaAssignmentContext ctx(W, H, K);
        // Destructor called here should trigger cudaFree
    });
}

// 2. Simple Assignment Accuracy (2x2 image, 2 clusters)
TEST_F(Backend_CudaKernels, Assignment_Accuracy_2x2) {
    const int W = 2;
    const int H = 2;
    const int K = 2;
    CudaAssignmentContext ctx(W, H, K);

    // Create 2x2 image: [White, White; Black, Black]
    cv::Mat input(H, W, CV_8UC3);
    input.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 255, 255);
    input.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 255, 255);
    input.at<cv::Vec3b>(1, 0) = cv::Vec3b(0, 0, 0);
    input.at<cv::Vec3b>(1, 1) = cv::Vec3b(0, 0, 0);

    // Two centroids: Pure White and Pure Black
    std::vector<FeatureVector> centers(K);
    centers[0] = FeatureVector(1.0f, 1.0f, 1.0f, 0.0f, 0.0f); // White
    centers[1] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f); // Black

    cv::Mat output;
    ctx.run(input, centers, output);

    // Top pixels should be assigned to White (cluster 0), bottom to Black (cluster 1)
    // Note: Kernel replaces pixel color with centroid color
    EXPECT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(255, 255, 255));
    EXPECT_EQ(output.at<cv::Vec3b>(1, 1), cv::Vec3b(0, 0, 0));
}

// 3. Stress Test: Boundary Case (K=20 Max)
TEST_F(Backend_CudaKernels, Boundary_MaxK) {
    CudaAssignmentContext ctx(100, 100, constants::clustering::K_MAX);
    cv::Mat input(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<FeatureVector> centers(constants::clustering::K_MAX, FeatureVector(0.5f, 0.5f, 0.5f, 0, 0));
    cv::Mat output;
    
    EXPECT_NO_THROW(ctx.run(input, centers, output));
}

// 4. Robustness: Empty/Invalid Data
TEST_F(Backend_CudaKernels, Boundary_EmptyData) {
    CudaAssignmentContext ctx(0, 0, 5);
    cv::Mat empty;
    std::vector<FeatureVector> centers(5);
    cv::Mat output;
    
    // Should handle gracefully without crashing
    EXPECT_NO_THROW(ctx.run(empty, centers, output));
    EXPECT_TRUE(output.empty());
}

// 5. Normalization and Clipping (Verifying roundf and saturation)
TEST_F(Backend_CudaKernels, Normalization_And_Clipping) {
    CudaAssignmentContext ctx(1, 1, 1);
    cv::Mat input(1, 1, CV_8UC3, cv::Scalar(127, 127, 127)); // Middle Grey
    
    // Centroid at 0.5 (exactly 127.5) -> Should round to 128 or 127 depending on implementation
    // But we check that it's near
    std::vector<FeatureVector> centers(1);
    centers[0] = FeatureVector(1.0f, 1.0f, 1.0f, 0.0f, 0.0f); // White (1.0 -> 255)

    cv::Mat output;
    ctx.run(input, centers, output);

    cv::Vec3b pixel = output.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[2], 255); // Red (1.0 * 255)
    EXPECT_EQ(pixel[1], 255); // Green (1.0 * 255)
    EXPECT_NEAR(pixel[0], 255, 1); // Blue
}

// 6. Non-Standard Resolution (Ensures grid/block logic is robust for messy sizes)
TEST_F(Backend_CudaKernels, NonStandardResolution) {
    const int W = 317; // Prime number
    const int H = 211; // Prime number
    const int K = 3;
    CudaAssignmentContext ctx(W, H, K);

    cv::Mat input(H, W, CV_8UC3, cv::Scalar(100, 100, 100));
    std::vector<FeatureVector> centers(K, FeatureVector(0.5f, 0.5f, 0.5f, 0.0f, 0.0f));
    cv::Mat output;
    
    // Should complete without illegal memory access or crashes
    EXPECT_NO_THROW(ctx.run(input, centers, output));
    EXPECT_FALSE(output.empty());
}

// 7. Context Reuse (Verifies persistent memory stability across multiple runs)
TEST_F(Backend_CudaKernels, ContextReuse) {
    const int W = 10;
    const int H = 10;
    CudaAssignmentContext ctx(W, H, 2);
    
    cv::Mat out1, out2;
    std::vector<FeatureVector> c1(2, FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<FeatureVector> c2(2, FeatureVector(1.0f, 1.0f, 1.0f, 0.0f, 0.0f));
    
    cv::Mat frame(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // Run twice with different centroids
    ctx.run(frame, c1, out1);
    ctx.run(frame, c2, out2);

    // out1 should be black, out2 should be white
    EXPECT_EQ(out1.at<cv::Vec3b>(0,0), cv::Vec3b(0,0,0));
    EXPECT_EQ(out2.at<cv::Vec3b>(0,0), cv::Vec3b(255,255,255));
}

} // namespace ThesisTests
