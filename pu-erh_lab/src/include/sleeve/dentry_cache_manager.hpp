#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>

#include "sleeve_element/sleeve_element.hpp"
#include "type/type.hpp"

#pragma once

namespace puerhlab {
class DCacheManager {
 private:
  std::unordered_map<sl_path_t, std::shared_ptr<SleeveElement>> _cache;
  std::map<uint32_t, sl_path_t>                                 _access_history;
  uint32_t                                                      _capacity;
  uint32_t                                                      _access_counter;

 public:
  explicit DCacheManager();
  explicit DCacheManager(uint32_t capacity);

  void RecordAccess(const sl_path_t &path);
  auto AccessElement(const sl_path_t &path);
};
};  // namespace puerhlab