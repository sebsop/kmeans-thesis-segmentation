#include "clustering/preprocessors/rcc_data_preprocessor.hpp"

#include "common/utils.hpp"

namespace kmeans::clustering {

cv::Mat RccDataPreprocessor::prepare(const cv::Mat& frame) {
    core::Coreset newCoreset = core::buildCoresetFromFrame(frame);

    if (m_frameCount % m_rebuildInterval == 0) {
        m_currentCoreset = newCoreset;
    } else {
        m_currentCoreset = core::mergeCoresets(m_currentCoreset, newCoreset);
    }

    m_frameCount++;

    int n = static_cast<int>(m_currentCoreset.points.size());
    cv::Mat samples(n, 5, CV_32F);

    for (int i = 0; i < n; ++i) {
        const auto& pt = m_currentCoreset.points[i];
        cv::Vec<float, 5> feature = common::makeFeature(pt.bgr, pt.x, pt.y);

        float* row = samples.ptr<float>(i);
        for (int d = 0; d < 5; ++d) {
            row[d] = feature[d];
        }
    }

    return samples;
}

void RccDataPreprocessor::reset() {
    m_frameCount = 0;
    m_currentCoreset.points.clear();
}

} // namespace kmeans::clustering
