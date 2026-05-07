/**
 * @file clustering_manager.hpp
 * @brief High-level controller for the clustering pipeline.
 */

#pragma once

#include <memory>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "common/constants.hpp"

namespace kmeans::clustering {
class KMeansEngine;

/**
 * @class ClusteringManager
 * @brief The central orchestrator for the video segmentation system.
 *
 * This class acts as a Facade, coordinating the three major stages of the
 * pipeline:
 * 1. Data Preprocessing (sampling pixels and building feature vectors).
 * 2. Centroid Optimization (running the K-Means loop via a KMeansEngine).
 * 3. Real-time Segmentation (assigning full-resolution frames via a CudaAssignmentContext).
 *
 * It manages the global configuration (SegmentationConfig) and the lifecycle
 * of all strategy components. It uses the Pimpl idiom to hide complex CUDA/OpenCV
 * implementation details from the rest of the application.
 */
class ClusteringManager {
  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Pimpl handle for internal state management

  public:
    /** @brief Initializes the manager with default configuration. */
    ClusteringManager();

    /** @brief Cleans up all managed engines and contexts. */
    ~ClusteringManager();

    /** @brief Returns a mutable reference to the application's clustering configuration. */
    [[nodiscard]] common::SegmentationConfig& getConfig() noexcept;

    /** @brief Returns a read-only reference to the application's clustering configuration. */
    [[nodiscard]] const common::SegmentationConfig& getConfig() const noexcept;

    /** @brief Retrieves the current set of optimized centroids. */
    [[nodiscard]] const std::vector<FeatureVector>& getCenters() const noexcept;

    /** @brief Provides direct access to the currently active execution engine. */
    [[nodiscard]] KMeansEngine* getEngine() const noexcept;

    /**
     * @brief Re-instantiates the engine and initializer based on the current config.
     * Call this after modifying algorithm selection in the UI.
     */
    void updateStategyImplementations();

    /** @brief Clears the current centroids, forcing a re-initialization. */
    void resetCenters();

    /** @brief Manually sets the cluster centers (used for testing or pre-seeded results). */
    void setInitialCenters(std::span<const FeatureVector> centers);

    /**
     * @brief Uses the active Initializer strategy to find starting points.
     * @param frame The frame from which to sample initial points.
     */
    [[nodiscard]] std::vector<FeatureVector> generateInitialCenters(const cv::Mat& frame);

    /**
     * @brief Performs full-resolution segmentation on the given frame.
     * Uses the high-performance GPU assignment path.
     * @return A labeled image (cv::Mat) where each pixel represents a cluster ID.
     */
    [[nodiscard]] cv::Mat segmentFrame(const cv::Mat& frame);

    /**
     * @brief Runs the full K-Means optimization loop to find the best centroids.
     * This is the "Learning" phase of the algorithm.
     * @return The newly computed centroids.
     */
    [[nodiscard]] std::vector<FeatureVector> computeCenters(const cv::Mat& frame);
};

} // namespace kmeans::clustering