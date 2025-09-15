#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "pipeline_utils.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
using FrameId = uint64_t;
enum class EntryState : uint8_t { NotStarted = 0, InProgress, Done, Cancelled };

struct CacheEntry {
  std::shared_ptr<ImageBuffer>                                _input;
  std::shared_ptr<ImageBuffer>                                _output;
  std::atomic<EntryState>                                     _state{EntryState::NotStarted};

  // Use promise/future to let other threads wait for the result
  std::shared_ptr<std::promise<std::shared_ptr<ImageBuffer>>> _promise;
  std::shared_future<std::shared_ptr<ImageBuffer>>            _shared_future;

  size_t                                                      params_hash = 0;
  std::chrono::steady_clock::time_point                       last_access;

  CacheEntry() {
    _promise       = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
    _shared_future = _promise->get_future().share();
    last_access    = std::chrono::steady_clock::now();
  }
};

class PreviewPipelineStage {
 private:
  std::mutex                                     _map_mutex;
  LRUCache<FrameId, std::shared_ptr<CacheEntry>> _cache_map;

  size_t                                         _max_cache_size = 5;  // Max 5 entries in cache

  std::map<OperatorType, OperatorEntry>          _operators;

  PreviewPipelineStage*                          next_stage = nullptr;

 public:
  PipelineStageName _stage;
  PreviewPipelineStage() = delete;
  PreviewPipelineStage(PipelineStageName stage);
  void SetOperator(OperatorType, nlohmann::json& param);
  void EnableOperator(OperatorType, bool enable);

  void SetNextStage(PreviewPipelineStage* next);

  auto GetStageNameString() const -> std::string;

  auto ApplyStage(FrameId id, std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<CacheEntry>;
};
};  // namespace puerhlab
