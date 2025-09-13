#include "edit/pipeline/pipeline_utils.hpp"

#include <easy/profiler.h>

#include <memory>
#include <stdexcept>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
PipelineStage::PipelineStage(PipelineStageName stage, bool on_gpu)
    : _on_gpu(on_gpu), _stage(stage) {
  if (_stage == PipelineStageName::Image_Loading) {
    _input_cache_valid = true;  // No input for image loading stage, so input cache is always valid
  }
}

void PipelineStage::SetNeighbors(PipelineStage* prev, PipelineStage* next) {
  prev_stage = prev;
  next_stage = next;
}

void PipelineStage::SetOperator(OperatorType op_type, nlohmann::json& param) {
  auto it = _operators.find(op_type);
  if (it == _operators.end()) {
    _operators.emplace(op_type,
                       OperatorEntry{true, OperatorFactory::Instance().Create(op_type, param)});
    SetOutputCacheValid(false);
  } else {
    (it->second)._op->SetParams(param);
    SetOutputCacheValid(false);
  }
}

void PipelineStage::EnableOperator(OperatorType op_type, bool enable) {
  auto it = _operators.find(op_type);
  if (it != _operators.end()) {
    if (it->second._enable != enable) {
      SetOutputCacheValid(false);
    }
    it->second._enable = enable;
  }
}

void PipelineStage::SetInputImage(std::shared_ptr<ImageBuffer> input) {
  _input_img = input;
  _input_set = true;
}

void PipelineStage::SetInputCacheValid(bool valid) {
  _input_cache_valid = valid;
  if (!valid) {
    // Invalidate output cache if input cache is invalidated
    SetOutputCacheValid(false);
  }
}

void PipelineStage::SetOutputCacheValid(bool valid) {
  if (_output_cache_valid != valid && !valid) {
    _output_cache.reset();
    if (next_stage) {
      next_stage->SetInputCacheValid(false);
    }
  }
  _output_cache_valid = valid;
}

auto PipelineStage::ApplyStage() -> std::shared_ptr<ImageBuffer> {
  if (!_input_set) {
    throw std::runtime_error("PipelineExecutor: No valid input image set");
  }
  if (_on_gpu) {
    throw std::runtime_error("PipelineExecutor: GPU processing not implemented");
  }

  if (CacheValid()) {
    if (next_stage && next_stage->CacheValid()) {
      // Both input and output cache are valid, skip processing and copying
      return nullptr;
    }
    // Output cache is valid, but next stage's input cache is not valid, copy output cache to next
    return _output_cache;
  }

  bool has_enabled_op = _operators.size() > 0;
  if (has_enabled_op) {
    ImageBuffer current_img = _input_img->Clone();
    for (const auto& op_entry : _operators) {
      if (op_entry.second._enable) {
        EASY_NONSCOPED_BLOCK(
            std::format("Apply Operator: {}", op_entry.second._op->GetScriptName()).c_str(),
            profiler::colors::Cyan);
        current_img = op_entry.second._op->Apply(current_img);
        EASY_END_BLOCK;
      }
    }
    _output_cache = std::make_shared<ImageBuffer>(std::move(current_img));
  } else {
    _output_cache = _input_img;
  }

  // _input_set = false;
  // _input_img.reset();
  SetOutputCacheValid(true);
  if (next_stage) {
    next_stage->SetInputCacheValid(true);
  }
  return _output_cache;
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