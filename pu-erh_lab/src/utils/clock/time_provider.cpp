//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "utils/clock/time_provider.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>

namespace puerhlab {
std::atomic<std::chrono::system_clock::time_point> TimeProvider::_cached_sys_time;
std::atomic<std::chrono::steady_clock::time_point> TimeProvider::_cached_steady_time;

void                                               TimeProvider::Refresh() {
  _cached_sys_time = std::chrono::system_clock::now();
  _cached_steady_time = std::chrono::steady_clock::now();
}

std::chrono::system_clock::time_point TimeProvider::Now() {
  auto elapsed = std::chrono::steady_clock::now() - _cached_steady_time.load();
  return _cached_sys_time.load() +
         std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed);
}

std::string TimeProvider::TimePointToString(const std::chrono::system_clock::time_point& tp) {
  std::time_t        t  = std::chrono::system_clock::to_time_t(tp);
  std::tm            tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
  return oss.str();
}
}  // namespace puerhlab