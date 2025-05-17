#include "storage/image_pool/image_pool_manager.hpp"

#include <memory>
#include <optional>

#include "type/type.hpp"

namespace puerhlab {
ImagePoolManager::ImagePoolManager()
    : _capacity_thumb(_default_capacity_thumb), _capacity_full(_default_capacity_full) {}

ImagePoolManager::ImagePoolManager(uint32_t capacity_thumb, uint32_t capacity_full)
    : _capacity_thumb(capacity_thumb), _capacity_full(capacity_full) {}

void ImagePoolManager::Insert(const std::shared_ptr<Image> img) { _image_pool[img->_image_id] = img; }

auto ImagePoolManager::PoolContains(const image_id_t &id) -> bool { return _image_pool.contains(id); }

auto ImagePoolManager::AccessElement(const image_id_t &id, const AccessType type)
    -> std::optional<std::weak_ptr<Image>> {
  switch (type) {
    case AccessType::FULL_IMG: {
      auto it = _cache_map_full.find(id);
      if (it == _cache_map_full.end()) {
        return std::nullopt;
      }
      // LRU, place the most recent record to the front
      _cache_list_full.splice(_cache_list_full.begin(), _cache_list_full, it->second);
      return _image_pool[it->first];
    }
    case AccessType::THUMB: {
      auto it = _cache_map_thumb.find(id);
      if (it == _cache_map_thumb.end()) {
        return std::nullopt;
      }
      // LRU, place the most recent record to the front
      _cache_list_full.splice(_cache_list_thumb.begin(), _cache_list_thumb, it->second);
      return _image_pool[it->first];
    }
    case AccessType::META:
      return _image_pool.at(id);
  }
  return std::nullopt;
}

void ImagePoolManager::RecordAccess(const image_id_t &id, const AccessType type) {
  switch (type) {
    case AccessType::FULL_IMG: {
      auto it = _cache_map_full.find(id);
      if (it == _cache_map_full.end()) {
        if (_cache_list_full.size() >= _capacity_full) {
          Evict(type);
        }
        // Place the new-added record to the front
        _cache_list_full.push_front(id);
        _cache_map_full[id] = _cache_list_full.begin();
      } else {
        _cache_list_full.splice(_cache_list_full.begin(), _cache_list_full, it->second);
        if (_cache_list_full.front() != id) {
          _cache_list_full.front() = id;
        }
        _with_full.insert(id);
      }
      break;
    }
    case AccessType::THUMB: {
      auto it = _cache_map_thumb.find(id);
      if (it == _cache_map_thumb.end()) {
        if (_cache_list_thumb.size() >= _capacity_thumb) {
          Evict(type);
        }
        _cache_list_thumb.push_front(id);
        _cache_map_thumb[id] = _cache_list_thumb.begin();
      } else {
        _cache_list_thumb.splice(_cache_list_thumb.begin(), _cache_list_thumb, it->second);
        if (_cache_list_thumb.front() != id) {
          _cache_list_thumb.front() = id;
        }
        _with_thumb.insert(id);
      }
      break;
    }
    case AccessType::META: {
    }
  }
}

void ImagePoolManager::RemoveRecord(const image_id_t &id, const AccessType type) {
  switch (type) {
    case AccessType::FULL_IMG: {
      auto it = _cache_map_full.find(id);
      if (it != _cache_map_full.end()) {
        _cache_list_full.erase(it->second);
        _cache_map_full.erase(it);
        _with_full.erase(it->second);
      }
      break;
    }
    case AccessType::THUMB: {
      auto it = _cache_map_thumb.find(id);
      if (it != _cache_map_thumb.end()) {
        _cache_list_thumb.erase(it->second);
        _cache_map_thumb.erase(it);
        _with_thumb.erase(it->second);
      }
      break;
    }
    case AccessType::META: {
    }
  }
}

auto ImagePoolManager::Evict(const AccessType type) -> std::optional<std::weak_ptr<Image>> {
  switch (type) {
    case AccessType::FULL_IMG: {
      if (_cache_list_full.empty()) {
        return std::nullopt;
      }
      auto last = _cache_list_full.end();
      --last;
      _cache_map_full.erase(*last);
      _with_full.erase(*last);
      auto evicted_img = _image_pool[*last];
      _cache_list_full.pop_back();
      return evicted_img;
    }
    case AccessType::THUMB: {
      if (_cache_list_thumb.empty()) {
        return std::nullopt;
      }
      auto last = _cache_list_thumb.end();
      --last;
      _cache_map_thumb.erase(*last);
      _with_thumb.erase(*last);
      auto evicted_img = _image_pool[*last];
      _cache_list_thumb.pop_back();
      return evicted_img;
    }
    case AccessType::META: {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

auto ImagePoolManager::CacheContains(const image_id_t &id, const AccessType type) -> bool {
  switch (type) {
    case AccessType::FULL_IMG: {
      return _cache_map_full.contains(id);
    }
    case AccessType::THUMB: {
      return _cache_map_thumb.contains(id);
    }
    case AccessType::META: {
      return _image_pool.contains(id);
    }
  }
  return false;
}

};  // namespace puerhlab