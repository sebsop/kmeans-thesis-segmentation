#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>

#include "clustering/clustering_manager.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering {

using namespace kmeans;
using namespace kmeans::clustering;

class Clustering_Manager : public ::testing::Test {
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

// 1. Integration: Full Segmentation Pipeline
TEST_F(Clustering_Manager, FullPipelineExecution) {
    ClusteringManager manager;
    auto& config = manager.getConfig();
    config.k = 2;
    config.algorithm = kmeans::common::AlgorithmType::KMEANS_REGULAR;
    
    // 64x64 frame with two distinct colors
    cv::Mat frame(64, 64, CV_8UC3);
    frame(cv::Rect(0, 0, 32, 64)).setTo(cv::Scalar(255, 0, 0));  // Blue
    frame(cv::Rect(32, 0, 32, 64)).setTo(cv::Scalar(0, 0, 255)); // Red
    
    cv::Mat result = manager.segmentFrame(frame);
    
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows, frame.rows);
    EXPECT_EQ(result.cols, frame.cols);
    
    auto centers = manager.getCenters();
    EXPECT_EQ(centers.size(), 2);
}

// 2. Hot-Swapping Strategy
TEST_F(Clustering_Manager, AlgorithmHotSwapping) {
    ClusteringManager manager;
    auto& config = manager.getConfig();
    
    config.algorithm = kmeans::common::AlgorithmType::KMEANS_REGULAR;
    cv::Mat frame(16, 16, CV_8UC3, cv::Scalar(0, 255, 0));
    
    (void)manager.segmentFrame(frame);
    auto* engine1 = manager.getEngine();
    
    // Switch to Quantum
    config.algorithm = kmeans::common::AlgorithmType::KMEANS_QUANTUM;
    (void)manager.segmentFrame(frame);
    auto* engine2 = manager.getEngine();
    
    EXPECT_NE(engine1, engine2);
}

// 3. Temporal Coherence (Center Reuse)
TEST_F(Clustering_Manager, CenterPersistenceBetweenFrames) {
    ClusteringManager manager;
    manager.getConfig().k = 3;
    manager.getConfig().learningInterval = 5; // Only re-cluster every 5 frames
    
    cv::Mat frame(16, 16, CV_8UC3, cv::Scalar(100, 100, 100));
    
    // Frame 0: Clusters
    auto centers0 = manager.computeCenters(frame);
    
    // Frame 1: Should reuse centers0 (learningInterval=5)
    auto centers1 = manager.computeCenters(frame);
    
    for(int i=0; i<3; ++i) {
        for(int d=0; d<5; ++d) {
            EXPECT_EQ(centers0[i][d], centers1[i][d]);
        }
    }
}

// 4. Initial Center Override
TEST_F(Clustering_Manager, SetInitialCentersOverride) {
    ClusteringManager manager;
    manager.getConfig().k = 2;
    
    std::vector<FeatureVector> forced(2);
    forced[0] = FeatureVector(0.1f, 0.1f, 0.1f, 0.1f, 0.1f);
    forced[1] = FeatureVector(0.9f, 0.9f, 0.9f, 0.9f, 0.9f);
    
    manager.setInitialCenters(forced);
    
    cv::Mat frame(16, 16, CV_8UC3, cv::Scalar(128, 128, 128));
    manager.getConfig().maxIterations = 0; // Don't move them
    
    auto centers = manager.computeCenters(frame);
    
    EXPECT_NEAR(centers[0][0], 0.1f, 1e-5);
    EXPECT_NEAR(centers[1][0], 0.9f, 1e-5);
}

} // namespace ThesisTests::Clustering
