#include "sleeve/dentry_cache_manager.hpp"

#include <cstdint>
#include <optional>

#include "type/type.hpp"

namespace puerhlab {
DCacheManager::DCacheManager() : _capacity(512) {}
DCacheManager::DCacheManager(uint32_t capacity) : _capacity(capacity) {}

/**
 * @brief Check if a path exists in the cache
 *
 * @param path
 * @return true
 * @return false
 */
auto DCacheManager::Contains(const sl_path_t& path) -> bool {
  return _cache_map.find(path) != _cache_map.end();
}

/**
 * @brief Access an element in the cache.
 *
 * @param path
 * @return std::optional<sl_element_id_t> an element_id to the element or null if path not presents
 * in the cache
 */
auto DCacheManager::AccessElement(const sl_path_t path) -> std::optional<sl_element_id_t> {
  auto it = _cache_map.find(path);
  if (it == _cache_map.end()) {
    return std::nullopt;
  }
  _cache_list.splice(_cache_list.begin(), _cache_list, it->second);
  return it->second->second;
}

void DCacheManager::RecordAccess(const sl_path_t path, const sl_element_id_t element_id) {
  auto it = _cache_map.find(path);
  if (it != _cache_map.end()) {
    _cache_list.splice(_cache_list.begin(), _cache_list, it->second);
    if (_cache_list.front().second != element_id) {
      _cache_list.front() = {path, element_id};
    }
  } else {
    if (_cache_list.size() >= _capacity) {
      Evict();
    }

    _cache_list.push_front({path, element_id});
    _cache_map[path] = _cache_list.begin();
  }
}

void DCacheManager::RemoveRecord(const sl_path_t path) {
  auto it = _cache_map.find(path);
  if (it != _cache_map.end()) {
    _cache_list.erase(it->second);
    _cache_map.erase(it);
  }
}

auto DCacheManager::Evict() -> std::optional<sl_element_id_t> {
  if (_cache_list.empty()) {
    return std::nullopt;
  }
  auto last = _cache_list.end();
  --last;
  _cache_map.erase(last->first);
  auto evicted_id = last->second;
  _cache_list.pop_back();
  ++_evict_count;
  if (_access_count != 0 && (double)_evict_count / (double)_access_count > 0.8) {
    Resize(_capacity * 1.2);
  }
  return evicted_id;
}

void DCacheManager::Resize(uint32_t new_capacity) {
  if (new_capacity > _capacity) {
    Flush();
  }
  _capacity = new_capacity;
}

void DCacheManager::Flush() {
  _cache_map.clear();
  _cache_list.clear();
}
};  // namespace puerhlab