/**
 * @file benchmark_observer.hpp
 * @brief Observer interface for benchmark completion events.
 */

#pragma once

#include "io/benchmark_result.hpp"

namespace kmeans::io {

/**
 * @class IBenchmarkObserver
 * @brief Interface for components that need to react to benchmark results.
 *
 * Any class that implements this interface can register itself with the
 * BenchmarkRunner to receive an asynchronous callback when a comparison
 * test finishes. This is primarily used by the UIManager to trigger the
 * display of the results overlay.
 */
class IBenchmarkObserver {
  public:
    virtual ~IBenchmarkObserver() = default;

    /**
     * @brief Called by the BenchmarkRunner when all engines have finished.
     * @param result The aggregated metrics and frames from the benchmark.
     */
    virtual void onBenchmarkComplete(const BenchmarkComparisonResult& result) = 0;
};

} // namespace kmeans::io
