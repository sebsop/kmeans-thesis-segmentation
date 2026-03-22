#include "clustering/ClusteringManager.hpp"
#include "common/enums.hpp"
#include "common/config.hpp"
#include "common/constants.hpp"
#include "coreset.hpp"
#include "opencv2/core.hpp"

namespace kmeans {
    cv::Mat ClusteringManager::segmentFrame(const cv::Mat& frame) {
        std::vector<cv::Vec<float, 5>> centers = computeCenters(frame);

        if (!m_cudaContext || m_cudaContext->getWidth() != frame.cols || m_cudaContext->getK() != m_config.k) {
            m_cudaContext = std::make_unique<CudaAssignmentContext>(frame.cols, frame.rows, m_config.k);
        }

        cv::Mat result(frame.rows, frame.cols, CV_8UC3);

        m_cudaContext->run(
            frame,
            centers,
            result
        );

        return result;
    }

    std::vector<cv::Vec<float, 5>> ClusteringManager::computeCenters(const cv::Mat& frame) {
        bool shouldUpdate = (m_frameCount % m_config.learningInterval == 0) || !m_hasPrevious;
        m_frameCount++;

        if (!shouldUpdate && m_hasPrevious) {
            return m_previousCenters;
        }

        cv::Mat samples = prepareData(frame);

        std::vector<cv::Vec<float, 5>> initialCenters = selectInitialCenters(samples);

        std::vector<cv::Vec<float, 5>> finalCenters = executeClustering(samples, initialCenters);

        m_previousCenters = finalCenters;
        m_hasPrevious = true;

        return m_previousCenters;
    }

    cv::Mat ClusteringManager::prepareData(const cv::Mat& frame) {
        if (m_config.strategy == DataStrategy::RCC_TREES) {
            // Build the leaf from current frame
            Coreset leaf = buildCoresetFromFrame(frame);

            // Insert into the tree (this maintains the 'Cached' part of RCC)
            m_rcc.insertLeaf(leaf, SAMPLE_COUNT);

            // Get the top-level summary
            Coreset rootCoreset = m_rcc.getRootCoreset();

            // Convert the CoresetPoints into our standard 5D Matrix
            return convertCoresetToMat(rootCoreset);
        }
        else {
            // If strategy is FULL_DATA, just flatten the whole image
            return flattenFrameTo5D(frame);
        }
    }

    cv::Mat ClusteringManager::convertCoresetToMat(const Coreset& coreset) {
        int rows = static_cast<int>(coreset.points.size());
        cv::Mat samples(rows, 5, CV_32F);

        for (int i = 0; i < rows; ++i) {
            const auto& p = coreset.points[i];
            samples.at<float>(i, 0) = p.bgr[0] * COLOR_SCALE;
            samples.at<float>(i, 1) = p.bgr[1] * COLOR_SCALE;
            samples.at<float>(i, 2) = p.bgr[2] * COLOR_SCALE;
            samples.at<float>(i, 3) = p.x * SPATIAL_SCALE;
            samples.at<float>(i, 4) = p.y * SPATIAL_SCALE;
        }
        return samples;
    }

    cv::Mat ClusteringManager::flattenFrameTo5D(const cv::Mat& frame) {
        int rows = frame.rows;
        int cols = frame.cols;
        int totalPixels = rows * cols;

        // Create a matrix where each row is [B, G, R, X, Y]
        cv::Mat samples(totalPixels, 5, CV_32F);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int idx = r * cols + c;
                cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);

                // Normalized coordinates (0.0 to 1.0)
                float x01 = static_cast<float>(c) / cols;
                float y01 = static_cast<float>(r) / rows;

                // Apply global scales from Constants.hpp
                samples.at<float>(idx, 0) = pixel[0] * kmeans::COLOR_SCALE;
                samples.at<float>(idx, 1) = pixel[1] * kmeans::COLOR_SCALE;
                samples.at<float>(idx, 2) = pixel[2] * kmeans::COLOR_SCALE;
                samples.at<float>(idx, 3) = x01 * kmeans::SPATIAL_SCALE;
                samples.at<float>(idx, 4) = y01 * kmeans::SPATIAL_SCALE;
            }
        }
        return samples;
    }

    std::vector<cv::Vec<float, 5>> ClusteringManager::selectInitialCenters(const cv::Mat& samples) {
        // If we are in the middle of a video stream and K hasn't changed, 
        // the best initialization is the previous frame's result.
        if (m_hasPrevious && m_previousCenters.size() == m_config.k) {
            return m_previousCenters;
        }

        // Otherwise, use the fresh initialization method chosen in config
        if (m_config.init == Initialization::KMEANS_PLUSPLUS) {
            return Initialization::pickKMeansPlusPlus(samples, m_config.k);
        }

        return Initialization::pickRandom(samples, m_config.k);
    }

    std::vector<cv::Vec<float, 5>> ClusteringManager::executeClustering(
        const cv::Mat& samples,
        const std::vector<cv::Vec<float, 5>>& initialCenters)
    {
        if (m_config.algorithm == Algorithm::KMEANS_QUANTUM) {
            return runQuantumKMeans(samples, initialCenters, m_config.k);
        }

        return runClassicalKMeans(samples, initialCenters, m_config.k);
    }
}