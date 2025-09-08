#include "edit/pipeline/pipeline_utils.hpp"

#include <stdexcept>

#include "edit/operators/operator_factory.hpp"

namespace puerhlab {
PipelineStage::PipelineStage(PipelineStageName stage, bool on_gpu)
    : _stage(stage), _on_gpu(on_gpu) {}

void PipelineStage::SetOperator(OperatorType op_type, nlohmann::json& param) {
  auto it = _op_map.find(op_type);
  if (it == _op_map.end()) {
    _operators.emplace_back(true, OperatorFactory::Instance().Create(op_type, param));
    _op_map[op_type] = _operators.begin();
  } else {
    (it->second)->_op->SetParams(param);
  }
}

void PipelineStage::EnableOperator(OperatorType op_type, bool enable) {
  auto it = _op_map.find(op_type);
  if (it != _op_map.end()) {
    it->second->_enable = enable;
  }
}

void PipelineStage::SetInputImage(ImageBuffer&& input) {
  _input_img = std::move(input);
  _input_set = true;
}

auto PipelineStage::ApplyStage() -> ImageBuffer {
  if (!_input_set) {
    throw std::runtime_error("PipelineExecutor PipelineStageName: No valid input image set");
  }
  if (_on_gpu) {
    throw std::runtime_error("PipelineExecutor PipelineStageName: GPU processing not implemented");
  }

  auto& output = _input_img;
  for (auto& op : _operators) {
    if (op._enable) {
      output = op._op->Apply(output);
    }
  }

  _input_set = false;
  return {std::move(output)};
}

auto PipelineStage::HasInput() -> bool { return _input_set; }
};  // namespace puerhlab