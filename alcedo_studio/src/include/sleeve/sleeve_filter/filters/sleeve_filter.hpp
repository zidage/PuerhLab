//  Copyright 2025-2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <string>

#include "type/type.hpp"

namespace alcedo {

enum class ElementOrder { ASC, DESC };

enum class FilterType { DATETIME, EXIF, DEFAULT };
class SleeveFilter {
 public:
  FilterType   type_;
  virtual void ResetFilter()                      = 0;
  virtual auto GetPredicate() -> std::wstring     = 0;
  virtual auto ToJSON() -> std::wstring           = 0;
  virtual void FromJSON(const std::wstring j_str) = 0;
};
};  // namespace alcedo