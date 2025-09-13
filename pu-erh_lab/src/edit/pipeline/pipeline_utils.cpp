#include "edit/pipeline/pipeline_utils.hpp"

#include <easy/profiler.h>

#include <stdexcept>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
PipelineStage::PipelineStage(PipelineStageName stage, bool on_gpu)
    : _on_gpu(on_gpu), _stage(stage) {}

void PipelineStage::SetOperator(OperatorType op_type, nlohmann::json& param) {
  auto it = _operators.find(op_type);
  if (it == _operators.end()) {
    _operators.emplace(op_type,
                       OperatorEntry{true, OperatorFactory::Instance().Create(op_type, param)});
  } else {
    (it->second)._op->SetParams(param);
  }
}

void PipelineStage::EnableOperator(OperatorType op_type, bool enable) {
  auto it = _operators.find(op_type);
  if (it != _operators.end()) {
    it->second._enable = enable;
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
    if (op.second._enable) {
      EASY_NONSCOPED_BLOCK(
          std::format("Apply Operator: {}", op.second._op->GetScriptName()).c_str(),
          profiler::colors::Cyan);
      output = op.second._op->Apply(output);
      EASY_END_BLOCK;
    }
  }

  _input_set = false;
  return {std::move(output)};
}

auto PipelineStage::HasInput() -> bool { return _input_set; }

auto PipelineStage::GetStageNameString() const -> std::string {
  switch (_stage) {
    case PipelineStageName::Image_Loading:
      return "Image Loading";
    case PipelineStageName::To_WorkingSpace:
      return "To Working Space";
    case PipelineStageName::Basic_Adjustment:
      return "Basic Adjustment";
    case PipelineStageName::Color_Adjustment:
      return "Color Adjustment";
    case PipelineStageName::Detail_Adjustment:
      return "Detail Adjustment";
    case PipelineStageName::Output_Transform:
      return "Output Transform";
    case PipelineStageName::Geometry_Adjustment:
      return "Geometry Adjustment";
    default:
      return "Unknown Stage";
  }
}
};  // namespace puerhlab