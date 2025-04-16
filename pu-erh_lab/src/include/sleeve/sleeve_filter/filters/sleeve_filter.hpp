#pragma once

#include <cstdint>
#include <string>

#include "type/type.hpp"

namespace puerhlab {

enum class ElementOrder { ASC, DESC };

class SleeveFilter {
 public:
  virtual void ResetFilter()                  = 0;
  virtual auto GetPredicate() -> std::wstring = 0;
  virtual auto Hash() -> hash_t               = 0;
};
};  // namespace puerhlab