//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <string>

#include "sleeve_filter.hpp"

namespace puerhlab {
template <typename T>
class RangeFilter : public SleeveFilter {
 public:
  virtual void SetRange(T range_low, T range_high) = 0;
};
};  // namespace puerhlab