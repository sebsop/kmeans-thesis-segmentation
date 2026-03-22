#include "clustering/rcc_data_preprocessor.hpp"
#include "common/constants.hpp"

namespace kmeans {
namespace clustering {

    cv::Mat RccDataPreprocessor::prepare(const cv::Mat& frame) {
        CV_Assert(!frame.empty());
        Coreset leaf = buildCoresetFromFrame(frame);
        
        m_rcc.insertLeaf(leaf, SAMPLE_COUNT);
        
        Coreset rootCoreset = m_rcc.getRootCoreset();
        
        return convertCoresetToMat(rootCoreset);
    }

    cv::Mat RccDataPreprocessor::convertCoresetToMat(const Coreset& coreset) {
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

}
}
