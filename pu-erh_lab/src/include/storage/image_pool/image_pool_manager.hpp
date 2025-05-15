#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "image/image.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImagePoolManager {
  using ListIterator = std::list<std::pair<image_id_t, std::shared_ptr<Image>>>::iterator;

 private:
  std::unordered_map<image_id_t, ListIterator>             _cache_map;
  std::list<std::pair<image_id_t, std::shared_ptr<Image>>> _cache_list;
  uint32_t                                                 _capacity;

  uint32_t                                                 _evict_count  = 0;
  uint32_t                                                 _access_count = 0;

  std::unordered_set<image_id_t>                           _with_thumb;
  std::unordered_set<image_id_t>                           _with_full;

 public:
  static const uint32_t _default_capacity = 256;
  explicit ImagePoolManager();
  explicit ImagePoolManager(uint32_t capacity);

  auto AccessElement(const image_id_t &id) -> std::optional<std::shared_ptr<Image>>;
  void RecordAccess(const image_id_t &id, const std::shared_ptr<Image> image);
  void RemoveRecord(const image_id_t &id);
  auto Evict() -> std::optional<std::shared_ptr<Image>>;
  auto Contains(const image_id_t &id) -> bool;

  void Flush();
  void Resize(uint32_t new_capacity);
};
};  // namespace puerhlab