//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "sleeve_filter.hpp"

namespace alcedo {
template <typename T>
class ValueFilter {
 public:
  virtual void SetValue(T value) = 0;
};
};  // namespace alcedo