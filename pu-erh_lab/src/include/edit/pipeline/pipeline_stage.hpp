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

 public:
  auto ExportOperatorParams() const -> nlohmann::json {
    nlohmann::json j;
    j["type"]   = _op->GetOperatorType();
    j["enable"] = _enable;
    j["params"] = _op->GetParams();
    return j;
  }

  auto ImportOperatorParams(const nlohmann::json& j) -> void {
    if (j.contains("enable")) _enable = j["enable"].get<bool>();
    if (j.contains("params")) _op->SetParams(j["params"]);
  }
};

class PipelineStage {
 private:
  std::unique_ptr<std::map<OperatorType, OperatorEntry>> _operators;
  bool                                                   _is_streamable = true;
  bool                                                   _vec_enabled   = false;
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

  /**
   * @brief Set the parameters for an operator with the given type in this stage.
   *
   * @param op_type
   * @param param
   * @return int 1 if a new operator is created and added (if this stage was merged into the
   * execution stage, should call SetExecutionStages() to rebuild the kernel stream), 0 if an
   * existing operator is updated.
   */
  auto SetOperator(OperatorType, nlohmann::json param) -> int;

  auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*>;
  auto GetAllOperators() const -> std::map<OperatorType, OperatorEntry>& { return *_operators; }
  void EnableOperator(OperatorType, bool enable);
  auto HasStreamableOps() const -> bool {
    return _is_streamable && _kernel_stream._kernels.size() > 0;
  }

  void SetNeighbors(PipelineStage* prev, PipelineStage* next) {
    _prev_stage = prev;
    _next_stage = next;
  }

  void ResetNeighbors() {
    _prev_stage = nullptr;
    _next_stage = nullptr;
  }

  void SetInputCacheValid(bool valid);
  void SetOutputCacheValid(bool valid);
  auto CacheValid() const -> bool {
    if (!_enable_cache) return false;
    if (!_prev_stage) return _output_cache_valid;
    return _input_cache_valid && _output_cache_valid;
  }

  auto GetOutputCache() const -> std::shared_ptr<ImageBuffer> { return _output_cache; }

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

  /**
   * @brief Reset this stage to initial state
   *
   */
  void ResetAll();

  /**
   * @brief Reset the cache of this stage
   *
   */
  void ResetCache();

  /**
   * @brief Export the parameters of this stage and its operators to JSON (serialize)
   *
   * @return nlohmann::json
   */
  auto ExportStageParams() const -> nlohmann::json;

  /**
   * @brief Import the parameters of this stage and its operators from JSON (deserialize)
   *
   * @param j
   */
  void ImportStageParams(const nlohmann::json& j);
};

}  // namespace puerhlab