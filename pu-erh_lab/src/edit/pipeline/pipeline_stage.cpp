#include "edit/pipeline/pipeline_stage.hpp"

#include <memory>
#include <stdexcept>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
PipelineStage::PipelineStage(PipelineStageName stage, bool enable_cache, bool is_streamable)
    : _enable_cache(enable_cache), _is_streamable(is_streamable) {
  _operators = std::make_unique<std::map<OperatorType, OperatorEntry>>();
  if (_stage == PipelineStageName::Image_Loading) {
    _input_cache_valid = true;  // No input for image loading stage, so input cache is always valid
  }
}

auto PipelineStage::SetOperator(OperatorType op_type, nlohmann::json& param) -> int {
  auto it = _operators->find(op_type);
  if (it == _operators->end()) {
    auto op = OperatorFactory::Instance().Create(op_type, param);
    _operators->emplace(op_type, OperatorEntry{true, op});
    SetOutputCacheValid(false);
    for (auto* dependent : _dependents) {
      dependent->SetOutputCacheValid(false);
    }
    return 1;
  } else {
    (it->second)._op->SetParams(param);
    SetOutputCacheValid(false);
    for (auto* dependent : _dependents) {
      dependent->SetOutputCacheValid(false);
    }
    if (op_type == OperatorType::CST || op_type == OperatorType::LMT) {
      // CST and LMT need to regenerate their CPU processors, so we need to call
      // SetExecutionStages()
      return 1;
    }
    return 0;
  }
}

auto PipelineStage::GetOperator(OperatorType op_type) const -> std::optional<OperatorEntry*> {
  auto it = _operators->find(op_type);
  if (it == _operators->end()) {
    return std::nullopt;
  }
  return &(it->second);
}

void PipelineStage::EnableOperator(OperatorType op_type, bool enable) {
  auto it = _operators->find(op_type);
  if (it != _operators->end()) {
    if (it->second._enable != enable) {
      SetOutputCacheValid(false);
      for (auto* dependent : _dependents) {
        dependent->SetInputCacheValid(false);
      }
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
    if (_next_stage) {
      _next_stage->SetInputCacheValid(false);
    }
  }
  _output_cache_valid = valid;
}

auto PipelineStage::ApplyStage() -> std::shared_ptr<ImageBuffer> {
  if (!_input_set) {
    throw std::runtime_error("PipelineExecutor: No valid input image set");
  }

  if (_enable_cache) {
    if (CacheValid()) {
      if (_next_stage && _next_stage->CacheValid()) {
        // Both input and output cache are valid, skip processing and copying
        return nullptr;
      }
      // Output cache is valid, but next stage's input cache is not valid, copy output cache to next
      return _output_cache;
    }

    bool has_enabled_op = _operators->size() > 0;
    if (has_enabled_op && !_is_streamable) {
      // Non-streamable stage with enabled operators, process the entire image at once
      auto current_img = std::make_shared<ImageBuffer>(_input_img->Clone());
      for (const auto& op_entry : *_operators) {
        if (op_entry.second._enable) {
          op_entry.second._op->Apply(current_img);
        }
      }
      _output_cache = current_img;
    } else if (_is_streamable) {
      if (!HasStreamableOps()) {
        _output_cache = _input_img;
      } else {
        // Streamable stage with enabled operators, use tile scheduler
        if (!_tile_scheduler) {
          // If tile scheduler is not set, create one
          SetTileScheduler();
        }
        _tile_scheduler->SetInputImage(_input_img);
        auto current_img = _tile_scheduler->ApplyOps();
        _output_cache    = current_img;
      }
    } else {
      _output_cache = _input_img;
    }

    // _input_set = false;
    // _input_img.reset();
    SetOutputCacheValid(true);
    if (_next_stage) {
      _next_stage->SetInputCacheValid(true);
    }
    return _output_cache;
  } else {
    // No caching, always process the image without using streaming
    bool has_enabled_op = _operators->size() > 0;
    if (has_enabled_op) {
      std::shared_ptr<ImageBuffer> current_img = _input_img;
      for (const auto& op_entry : *_operators) {
        if (op_entry.second._enable) {
          op_entry.second._op->Apply(current_img);
        }
      }
      _output_cache = current_img;
    } else {
      _output_cache = _input_img;
    }

    // Set to false to avoid using cache
    SetOutputCacheValid(false);
    if (_next_stage) {
      _next_stage->SetInputCacheValid(false);
    }
    return _output_cache;
  }
}

auto PipelineStage::HasInput() -> bool { return _input_set; }

void PipelineStage::ResetAll() {
  _operators->clear();
  _input_img.reset();
  _output_cache.reset();
  _input_set          = false;
  _input_cache_valid  = false;
  _output_cache_valid = false;
  _tile_scheduler.reset();
  _dependents.clear();
}

void PipelineStage::ResetCache() {
  _output_cache.reset();
  _output_cache_valid = false;
  if (_next_stage) {
    _next_stage->SetInputCacheValid(false);
  }
}

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
    case PipelineStageName::Merged_Stage:
      return "Merged Stage";
    default:
      return "Unknown Stage";
  }

  return "Unknown Stage";
}
};  // namespace puerhlab