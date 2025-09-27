#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
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

  PipelineStage*                        prev_stage = nullptr;
  PipelineStage*                        next_stage = nullptr;

  std::shared_ptr<ImageBuffer>          _input_img;
  std::shared_ptr<ImageBuffer>          _output_cache;

  bool                                  _input_cache_valid  = false;
  bool                                  _output_cache_valid = false;
  bool                                  _input_set          = false;
  bool                                  _on_gpu             = false;

  bool                                  _enable_cache       = true;

 public:
  PipelineStageName _stage;
  PipelineStage() = delete;
  PipelineStage(PipelineStageName stage, bool enable_cache);
  void SetOperator(OperatorType, nlohmann::json& param);
  auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*>;
  void EnableOperator(OperatorType, bool enable);
  void SetInputImage(std::shared_ptr<ImageBuffer>);

  void SetNeighbors(PipelineStage* prev, PipelineStage* next);

  void SetInputCacheValid(bool valid);
  void SetOutputCacheValid(bool valid);

  auto CacheValid() -> bool { return _input_cache_valid && _output_cache_valid; };

  auto GetStageNameString() const -> std::string;

  auto HasInput() -> bool;

  auto ApplyStage() -> std::shared_ptr<ImageBuffer>;
};
};  // namespace puerhlab