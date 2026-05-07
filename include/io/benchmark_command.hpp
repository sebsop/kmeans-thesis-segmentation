/**
 * @file benchmark_command.hpp
 * @brief Command Pattern implementation for asynchronous benchmarking.
 */

#pragma once

#include <future>

#include <opencv2/core.hpp>

#include "common/config.hpp"
#include "io/benchmark_result.hpp"

namespace kmeans::io {

/**
 * @class IBenchmarkCommand
 * @brief Interface for a task that can be queued and executed by the BenchmarkRunner.
 *
 * This interface allows the system to treat different types of benchmarking
 * operations (e.g., full test, re-computation) uniformly. It leverages
 * C++ futures to allow for non-blocking result retrieval.
 */
class IBenchmarkCommand {
  public:
    virtual ~IBenchmarkCommand() = default;

    /** @brief Triggers the start of the asynchronous computation. */
    virtual void execute() = 0;

    /** @brief Returns a handle to the future result of the computation. */
    virtual std::future<BenchmarkComparisonResult>& getFuture() = 0;

    /** @brief Returns the configuration associated with this command. */
    virtual const common::SegmentationConfig& getConfig() const = 0;
};

/**
 * @class RunBenchmarkCommand
 * @brief A concrete command that executes a full Classical vs. Quantum comparison.
 *
 * This class encapsulates the data (frame, config) and the logic needed
 * to run a head-to-head K-Means test in a background thread.
 */
class RunBenchmarkCommand : public IBenchmarkCommand {
  private:
    cv::Mat m_frame;                                 ///< The static image data to process
    common::SegmentationConfig m_config;             ///< Hyper-parameters for the run
    std::future<BenchmarkComparisonResult> m_future; ///< Async handle to the result

  public:
    /**
     * @brief Constructs a command with the necessary input data.
     * @param frame The frozen image to segment.
     * @param config The algorithm settings to use.
     */
    RunBenchmarkCommand(const cv::Mat& frame, const common::SegmentationConfig& config);

    /** @brief Launches the background threads for both K-Means engines. */
    void execute() override;

    /** @brief Implementation of future retrieval. */
    std::future<BenchmarkComparisonResult>& getFuture() override { return m_future; }

    /** @brief Implementation of config retrieval. */
    const common::SegmentationConfig& getConfig() const override { return m_config; }
};

} // namespace kmeans::io
