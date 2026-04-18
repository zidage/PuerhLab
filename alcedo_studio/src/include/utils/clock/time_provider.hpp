//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once
#include <atomic>
#include <chrono>

namespace alcedo {
class TimeProvider {
 public:
  static void                                  Refresh();
  static std::chrono::system_clock::time_point Now();
  static std::string TimePointToString(const std::chrono::system_clock::time_point& tp);

 private:
  static std::atomic<std::chrono::system_clock::time_point> cached_sys_time_;
  static std::atomic<std::chrono::steady_clock::time_point> cached_steady_time_;
};
};  // namespace alcedo