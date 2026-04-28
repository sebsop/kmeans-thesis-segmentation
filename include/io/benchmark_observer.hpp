#pragma once

#include "io/benchmark_result.hpp"

namespace kmeans::io {

class IBenchmarkObserver {
  public:
    virtual ~IBenchmarkObserver() = default;
    virtual void onBenchmarkComplete(const BenchmarkComparisonResult& result) = 0;
};

} // namespace kmeans::io
