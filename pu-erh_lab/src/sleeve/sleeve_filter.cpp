#include "sleeve/sleeve_filter.hpp"

#include <cstddef>
#include <unordered_map>

namespace puerhlab {

size_t FilterHasher::operator()(const SleeveFilter &f) const {
  // TODO: Placeholder, add Implementation
  return std::hash<int>{}(1);
}
};  // namespace puerhlab