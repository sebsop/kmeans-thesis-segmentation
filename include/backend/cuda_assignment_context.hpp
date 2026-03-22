#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "cuda_runtime.h"

namespace kmeans {
class CudaAssignmentContext {
 private:
  int m_width, m_height, m_k;
  cudaStream_t m_stream;
  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;
  float* d_centers = nullptr;

  size_t m_imgSize;
  size_t m_centersSize;

 public:
  CudaAssignmentContext(int width, int height, int k);
  ~CudaAssignmentContext();

  CudaAssignmentContext(const CudaAssignmentContext&) = delete;
  CudaAssignmentContext& operator=(const CudaAssignmentContext&) = delete;

  int getWidth() { return m_width; }
  int getK() { return m_k; }

  void run(const cv::Mat& frame, const std::vector<cv::Vec<float, 5>>& centers,
           cv::Mat& output);
};
}