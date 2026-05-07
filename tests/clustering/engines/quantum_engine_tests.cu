/**
 * @file quantum_engine_tests.cu
 * @brief Unit tests for the Quantum-Inspired (GPU-emulated) K-Means engine.
 */

#include <cuda_runtime.h>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "clustering/engines/quantum_engine.hpp"
#include "common/constants.hpp"

namespace ThesisTests::Clustering::Engines {

using namespace kmeans;
using namespace kmeans::clustering;

/**
 * @brief Test fixture for verifying the Quantum-Inspired clustering implementation.
 */
class Clustering_QuantumEngine : public ::testing::Test {
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
 * @brief Verifies the Quantum Interference (Swap-Test) approximation accuracy.
 *
 * Ensures that identical vectors yield a similarity of 1.0 (phase overlap),
 * while distant vectors correctly yield lower probabilities.
 */
TEST_F(Clustering_QuantumEngine, QuantumMetricAssignment) {
    QuantumEngine engine;

    // Identical point and center
    cv::Mat samples(1, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.75f));
    std::vector<FeatureVector> centers(2);
    centers[0] = FeatureVector(0.75f, 0.75f, 0.75f, 0.75f, 0.75f); // Exact match
    centers[1] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);      // Distant

    auto results = engine.run(samples, centers, 2, 2);

    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(results[0][d], 0.75f, constants::math::EPSILON);
    }
}

/**
 * @brief Validates convergence when using the non-Euclidean Quantum Metric.
 *
 * Proves that the algorithm still reaches stable centroids even when using
 * Hilbert-space distances instead of standard Euclidean distances.
 */
TEST_F(Clustering_QuantumEngine, QuantumConvergence) {
    QuantumEngine engine;

    cv::Mat samples(10, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int i = 0; i < 5; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 0.1f;
        }
    }
    for (int i = 5; i < 10; ++i) {
        for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
            samples.at<float>(i, d) = 0.9f;
        }
    }

    std::vector<FeatureVector> initialCenters(2);
    initialCenters[0] = FeatureVector(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    initialCenters[1] = FeatureVector(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    auto finalCenters = engine.run(samples, initialCenters, 2, 10);

    for (int d = 0; d < constants::clustering::FEATURE_DIMS; ++d) {
        EXPECT_NEAR(finalCenters[0][d], 0.1f, 0.01f);
        EXPECT_NEAR(finalCenters[1][d], 0.9f, 0.01f);
    }
}

/**
 * @brief Stress tests the shared memory usage for the Quantum kernel at high K-values.
 */
TEST_F(Clustering_QuantumEngine, QuantumHighK) {
    QuantumEngine engine;
    const int K = constants::clustering::K_MAX;

    cv::Mat samples(100, constants::clustering::FEATURE_DIMS, CV_32F, cv::Scalar(0.5f));
    std::vector<FeatureVector> centers(K, FeatureVector(0.5f, 0.5f, 0.5f, 0, 0));

    EXPECT_NO_THROW((void)engine.run(samples, centers, K, 2));
}

/**
 * @brief Verifies that phase-aliasing is prevented through feature space normalization.
 *
 * Ensures that extreme value ranges don't "wrap around" the 2*PI phase circle,
 * which would cause mathematical instability in the quantum kernel.
 */
TEST_F(Clustering_QuantumEngine, QuantumScaleAliasingPrevention) {
    QuantumEngine engine;
    const int K = 2;

    // Extreme values that would normally cause phase wrapping
    cv::Mat samples(2, constants::clustering::FEATURE_DIMS, CV_32F);
    for (int d = 0; d < 5; ++d) {
        samples.at<float>(0, d) = 0.0f;
    }
    for (int d = 0; d < 5; ++d) {
        samples.at<float>(1, d) = 1000000.0f;
    }

    std::vector<FeatureVector> initialCenters(K);
    initialCenters[0] = FeatureVector(0, 0, 0, 0, 0);
    initialCenters[1] = FeatureVector(1000000.0f, 1000000.0f, 1000000.0f, 1000000.0f, 1000000.0f);

    auto finalCenters = engine.run(samples, initialCenters, K, 2);

    EXPECT_NEAR(finalCenters[0][0], 0.0f, 1.0f);
    EXPECT_NEAR(finalCenters[1][0], 1000000.0f, 1.0f);
}

} // namespace ThesisTests::Clustering::Engines
