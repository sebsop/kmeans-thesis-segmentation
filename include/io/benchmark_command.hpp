#pragma once

#include <future>

#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "io/benchmark_result.hpp"

namespace kmeans::io {

class IBenchmarkCommand {
  public:
    virtual ~IBenchmarkCommand() = default;
    virtual void execute() = 0;
    virtual std::future<BenchmarkComparisonResult>& getFuture() = 0;
    virtual const common::SegmentationConfig& getConfig() const = 0;
};

class RunBenchmarkCommand : public IBenchmarkCommand {
  private:
    cv::Mat m_frame;
    common::SegmentationConfig m_config;
    std::future<BenchmarkComparisonResult> m_future;

  public:
    RunBenchmarkCommand(const cv::Mat& frame, const common::SegmentationConfig& config);

    void execute() override;

    std::future<BenchmarkComparisonResult>& getFuture() override { return m_future; }

    const common::SegmentationConfig& getConfig() const override { return m_config; }
};

} // namespace kmeans::io
