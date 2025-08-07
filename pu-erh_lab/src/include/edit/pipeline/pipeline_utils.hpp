#pragma once

#include <memory>
#include <unordered_map>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {

struct OperatorEntry {
  bool                           _enable = true;
  std::shared_ptr<IOperatorBase> _op;
};

class PipelineStage {
  using OpPos = std::list<OperatorEntry>::iterator;

 private:
  std::unordered_map<OperatorType, OpPos> _op_map;
  std::list<OperatorEntry>                _operators;

  ImageBuffer                             _input_img;
  bool                                    _input_set = false;
  bool                                    _on_gpu    = false;

 public:
  PipelineStageName _stage;
  PipelineStage() = delete;
  PipelineStage(PipelineStageName stage, bool on_gpu);
  void SetOperator(OperatorType, nlohmann::json& param);
  void EnableOperator(OperatorType, bool enable);
  void SetInputImage(ImageBuffer&& input);

  auto HasInput() -> bool;

  auto ApplyStage() -> ImageBuffer;
};
};  // namespace puerhlab