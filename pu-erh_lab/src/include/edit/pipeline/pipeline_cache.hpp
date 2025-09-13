#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "pipeline_utils.hpp"
#include "type/type.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
enum class PipelineCachePriority { Low, Medium, High };

struct PipelineCacheKey {
  PipelineStageName _stage;
  OperatorType      _op_type;
  p_hash_t          _params_hash;

 public:
  PipelineCacheKey(OperatorEntry& entry);

  bool operator==(const PipelineCacheKey& other) const {
    return _stage == other._stage && _op_type == other._op_type &&
           _params_hash == other._params_hash;
  }
};

struct PipelineCacheValue {
  std::shared_ptr<ImageBuffer> _image;
  bool _dirty = true;  // If true, the cache is invalid and should not be used
};
}  // namespace puerhlab

namespace std {
template <>
struct hash<puerhlab::PipelineCacheKey> {
  std::size_t operator()(const puerhlab::PipelineCacheKey& k) const noexcept {
    std::size_t h1 = std::hash<int>{}(static_cast<int>(k._stage));
    std::size_t h2 = std::hash<int>{}(static_cast<int>(k._op_type));
    std::size_t h3 = std::hash<p_hash_t>{}(k._params_hash);

    return (h1 ^ (h2 << 1)) ^ (h3 << 1);
  }
};
}  // namespace std

namespace puerhlab {
class PipelineCache {
 private:
  LRUCache<PipelineCacheKey, PipelineCacheValue> _high_priority_cache;
  LRUCache<PipelineCacheKey, PipelineCacheValue> _medium_priority_cache;
  LRUCache<PipelineCacheKey, PipelineCacheValue> _low_priority_cache;

  static constexpr size_t                        _high_priority_cache_size   = 1;
  static constexpr size_t                        _medium_priority_cache_size = 1;
  static constexpr size_t                        _low_priority_cache_size    = 2;

  void                                           EvictCache(PipelineCachePriority priority);

 public:
  PipelineCache();

  auto GetCache(const PipelineCacheKey& key) -> std::optional<PipelineCacheValue>;
  void SetCache(const PipelineCacheKey& key, const PipelineCacheValue& value,
                PipelineCachePriority priority);
  void InvalidateCache(const PipelineCacheKey& key);
  void FlushCache();
};
}  // namespace puerhlab