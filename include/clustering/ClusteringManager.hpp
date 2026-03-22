#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include "backend/CudaAssignmentContext.hpp"
#include "common/config.hpp"
#include "coreset.hpp"
#include "rcc.hpp"

namespace kmeans {
    class ClusteringManager {
    private:
        std::vector<cv::Vec<float, 5>> m_previousCenters;
        bool m_hasPrevious = false;
        int m_frameCount = 0;
        std::unique_ptr<CudaAssignmentContext> m_cudaContext;
        SegmentationConfig m_config;
        RCC m_rcc;

    public:
        SegmentationConfig& getConfig() { return m_config; }
        cv::Mat segmentFrame(const cv::Mat& frame);
        std::vector<cv::Vec<float, 5>> computeCenters(const cv::Mat& frame);
        cv::Mat prepareData(const cv::Mat& frame);
        cv::Mat ClusteringManager::convertCoresetToMat(const Coreset& coreset);
        cv::Mat ClusteringManager::flattenFrameTo5D(const cv::Mat& frame);
        std::vector<cv::Vec<float, 5>> selectInitialCenters(const cv::Mat& samples);
        std::vector<cv::Vec<float, 5>> executeClustering(const cv::Mat& samples, const std::vector<cv::Vec<float, 5>>& initialCenters);
    };
}