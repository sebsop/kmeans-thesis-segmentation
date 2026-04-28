#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "common/constants.hpp"
#include "cuda_runtime.h"

namespace kmeans::backend {

class CudaAssignmentContext {
  private:
    int m_width;
    int m_height;
    int m_k;
    cudaStream_t m_stream;
    unsigned char* m_d_input = nullptr;
    unsigned char* m_d_output = nullptr;
    float* m_d_centers = nullptr;

    unsigned char* m_h_input_pinned = nullptr;
    unsigned char* m_h_output_pinned = nullptr;
    float* m_h_centers_pinned = nullptr;

    size_t m_imgSize;
    size_t m_centersSize;

  public:
    CudaAssignmentContext(int width, int height, int k);
    ~CudaAssignmentContext() noexcept;

    CudaAssignmentContext(const CudaAssignmentContext&) = delete;
    CudaAssignmentContext& operator=(const CudaAssignmentContext&) = delete;

    [[nodiscard]] int getWidth() const noexcept { return m_width; }
    [[nodiscard]] int getK() const noexcept { return m_k; }

    void run(const cv::Mat& frame, const std::vector<cv::Vec<float, constants::FEATURE_DIMS>>& centers,
             cv::Mat& output);
};

} // namespace kmeans::backend