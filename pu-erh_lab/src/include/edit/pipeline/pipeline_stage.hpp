#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/tile_scheduler.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
struct OperatorEntry {
  bool                           _enable = true;
  std::shared_ptr<IOperatorBase> _op;

  bool                           operator<(const OperatorEntry& other) const {
    return _op->GetPriorityLevel() < other._op->GetPriorityLevel();
  }
};

class PipelineStage {
 private:
  std::unique_ptr<std::map<OperatorType, OperatorEntry>> _operators;
  bool                                                   _is_streamable = true;
  KernelStream                                           _kernel_stream;
  std::unique_ptr<TileScheduler>                         _tile_scheduler;

  PipelineStage*                                         _prev_stage         = nullptr;
  PipelineStage*                                         _next_stage         = nullptr;

  std::shared_ptr<ImageBuffer>                           _input_img          = nullptr;
  std::shared_ptr<ImageBuffer>                           _output_cache       = nullptr;

  bool                                                   _enable_cache       = false;
  bool                                                   _input_cache_valid  = false;
  bool                                                   _output_cache_valid = false;
  bool                                                   _input_set          = false;

  std::vector<PipelineStage*>                            _dependents;

 public:
  PipelineStageName _stage;
  PipelineStage()                           = delete;
  PipelineStage(const PipelineStage& other) = delete;

  PipelineStage(PipelineStageName stage, bool enable_cache, bool is_streamable);

  auto IsStreamable() const -> bool { return _is_streamable; }
  void SetTileScheduler() {
    _tile_scheduler = std::make_unique<TileScheduler>(_input_img, _kernel_stream);
  }

  void SetInputImage(std::shared_ptr<ImageBuffer>);

  void SetOperator(OperatorType, nlohmann::json& param);
  auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*>;
  auto GetAllOperators() const -> std::map<OperatorType, OperatorEntry>& { return *_operators; }
  void EnableOperator(OperatorType, bool enable);
  auto HasStreamableOps() const -> bool {
    return _is_streamable && _kernel_stream._kernels.size() > 0;
  }

  void SetNeighbors(PipelineStage* prev, PipelineStage* next);
  void SetInputCacheValid(bool valid);
  void SetOutputCacheValid(bool valid);
  auto CacheValid() const -> bool { return _input_cache_valid && _output_cache_valid; }

  /**
   * @brief Used to track merged stages dependent on this stage
   *
   * @param dependent
   */
  void AddDependent(PipelineStage* dependent) { _dependents.push_back(dependent); }
  void ResetDependents() { _dependents.clear(); }

  auto GetStageNameString() const -> std::string;

  auto HasInput() -> bool;

  auto ApplyStage() -> std::shared_ptr<ImageBuffer>;

  auto GetKernelStream() -> KernelStream& { return _kernel_stream; }

  void ResetAll();
  void ResetCache();
};

}  // namespace puerhlab