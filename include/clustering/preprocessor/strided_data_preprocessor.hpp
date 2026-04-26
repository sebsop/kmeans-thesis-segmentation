#pragma once

#include <memory>
#include <opencv2/core/mat.hpp>

namespace kmeans::clustering {

class StridedDataPreprocessor {
public:
    StridedDataPreprocessor() = default;
    ~StridedDataPreprocessor();

    // Standard CPU-bound prepare
    cv::Mat prepare(const cv::Mat& frame);
    // Resets GPU memory
    void reset();

    // Advanced: Prepares and leaves the result on the GPU.
    // Returns the device pointer and number of points extracted.
    float* prepareDevice(const cv::Mat& frame, int stride, int& outNumPoints);

    // Advanced: Downloads the previously computed GPU result
    cv::Mat download() const;

private:
    void uploadAndRun(const cv::Mat& frame, int stride);

    void* m_d_frame_data = nullptr; // uchar3 pointer
    float* m_d_samples = nullptr;   // 5D features
    int m_cached_n = 0;             // original frame pixels (to detect resizes)
    int m_extracted_points = 0;     // number of points extracted after stride
};

} // namespace kmeans::clustering
