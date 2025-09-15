#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "concurrency/thread_pool.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "pipeline_utils.hpp"
#include "type/type.hpp"
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

  p_hash_t                                                    params_hash = 0;
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

  auto                                           CurrentParamsHash() -> p_hash_t;

  ThreadPool*                                    _attached_pool = nullptr;

 public:
  PipelineStageName _stage;
  PreviewPipelineStage() = delete;
  PreviewPipelineStage(PipelineStageName stage);
  void SetOperator(OperatorType, nlohmann::json& param);
  void EnableOperator(OperatorType, bool enable);
  void SetInputImage(FrameId fid, std::shared_ptr<ImageBuffer> input);

  void SetNextStage(PreviewPipelineStage* next);

  void AttachThreadPool(ThreadPool* pool) { _attached_pool = pool; }

  auto Process(FrameId fid, const ImageBuffer& input) -> std::shared_future<ImageBuffer>;

  /**
   * @brief Get the Stage Name String object, for logging and debugging
   *
   * @return std::string
   */
  auto GetStageNameString() const -> std::string;

  auto ApplyStage(FrameId fid) -> std::shared_ptr<ImageBuffer>;
};
};  // namespace puerhlab
