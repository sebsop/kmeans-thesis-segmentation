#pragma once

#include <opencv2/imgproc.hpp>

#include "clustering/preprocessors/data_preprocessor.hpp"

namespace kmeans::clustering {

class FullDataPreprocessor final : public DataPreprocessor {
  public:
    FullDataPreprocessor() = default;
    ~FullDataPreprocessor();

    [[nodiscard]] cv::Mat prepare(const cv::Mat& frame) override final;
    void reset() override final;

  private:
    int m_cached_n = 0;
    void* m_d_frame_data = nullptr;
    void* m_d_samples = nullptr;
};

} // namespace kmeans::clustering
