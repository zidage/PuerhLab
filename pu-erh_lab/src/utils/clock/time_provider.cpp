/*
 * @file        pu-erh_lab/src/utils/clock/time_provider.cpp
 * @brief       Implementation of a unified time service to provide time stamp to different modules
 * @author      Yurun Zi
 * @date        2025-05-02
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "utils/clock/time_provider.hpp"

#include <chrono>

namespace puerhlab {
std::atomic<std::chrono::system_clock::time_point> TimeProvider::_cached_sys_time;
std::atomic<std::chrono::steady_clock::time_point> TimeProvider::_cached_steady_time;

void                                               TimeProvider::Refresh() {
  _cached_sys_time    = std::chrono::system_clock::now();
  _cached_steady_time = std::chrono::steady_clock::now();
}

std::chrono::system_clock::time_point TimeProvider::Now() {
  auto elapsed = std::chrono::steady_clock::now() - _cached_steady_time.load();
  return _cached_sys_time.load() + std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed);
}
}  // namespace puerhlab