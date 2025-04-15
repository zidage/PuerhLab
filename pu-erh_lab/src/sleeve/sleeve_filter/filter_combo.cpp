#include "sleeve/sleeve_filter/filter_combo.hpp"

#include <cstddef>
#include <unordered_map>

namespace puerhlab {

size_t FilterComboHasher::operator()(const FilterCombo &f) const {
  // TODO: Placeholder, add Implementation
  return std::hash<int>{}(1);
}
};  // namespace puerhlab