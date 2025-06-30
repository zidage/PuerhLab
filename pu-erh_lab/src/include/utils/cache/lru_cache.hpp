#pragma once

#include <concepts>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
namespace puerhlab {
template <typename K>
concept Hashable = std::copy_constructible<K> && std::equality_comparable<K> && requires(K key) {
  { std::hash<K>{}(key) } -> std::convertible_to<std::size_t>;
};
template <Hashable K, typename V>
class LRUCache {
  using ListIterator = std::list<std::pair<K, V>>::iterator;

 private:
  std::unordered_map<K, ListIterator> _cache_map;
  std::list<std::pair<K, V>>          _cache_list;
  uint32_t                            _capacity;

  uint32_t                            _evict_count  = 0;
  uint32_t                            _access_count = 0;

 public:
  static const uint32_t _default_capacity = 256;

  explicit LRUCache() : _capacity(_default_capacity) {}
  explicit LRUCache(uint32_t capacity) : _capacity(capacity) {}

  auto Contains(const K& key) -> bool { return _cache_map.contains(key); }

  auto AccessElement(const K& key) -> std::optional<V> {
    auto it = _cache_map.find(key);
    if (it == _cache_map.end()) {
      return std::nullopt;
    }
    _cache_list.splice(_cache_list.begin(), _cache_list, it->second);
    return it->second->second;
  }

  void RecordAccess(const K& key, const V& val) {
    auto it = _cache_map.find(key);
    if (it != _cache_map.end()) {
      _cache_list.splice(_cache_list.begin(), _cache_list, it->second);
      if (_cache_list.front().second != val) {
        _cache_list.front() = {key, val};
      }
    } else {
      if (_cache_list.size() >= _capacity) {
        Evict();
      }

      _cache_list.push_front({key, val});
      _cache_map[key] = _cache_list.begin();
    }
  }

  void RemoveRecord(const K& path) {
    auto it = _cache_map.find(path);
    if (it != _cache_map.end()) {
      _cache_list.erase(it->second);
      _cache_map.erase(it);
    }
  }

  auto Evict() -> std::optional<V> {
    if (_cache_list.empty()) {
      return std::nullopt;
    }
    auto last = _cache_list.end();
    --last;
    _cache_map.erase(last->first);
    auto evicted_id = last->second;
    _cache_list.pop_back();
    ++_evict_count;
    // Resize
    if (_access_count != 0 && (double)_evict_count / (double)_access_count > 0.8) {
      Resize(static_cast<uint32_t>(_capacity * 1.2));
    }
    return evicted_id;
  }

  void Resize(uint32_t new_capacity) {
    if (new_capacity > _capacity) {
      Flush();
    }
    _capacity = new_capacity;
  }

  void Flush() {
    _cache_map.clear();
    _cache_list.clear();
  }
};
};  // namespace puerhlab