#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <utility>

namespace kmeans::common {

class ScopedTimer {
  private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_start;

  public:
    explicit ScopedTimer(std::string name)
        : m_name(std::move(name)), m_start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
        std::cout << "[PROFILE] " << m_name << ": " << duration << " us\n";
    }
};

} // namespace kmeans::common
