#include "edit/pipeline/preview_pipeline_utils.hpp"

#include <memory>
#include <mutex>

#include "image/image_buffer.hpp"

namespace puerhlab {
PreviewPipelineStage::PreviewPipelineStage(PipelineStageName stage) : _stage(stage) {}

void PreviewPipelineStage::SetInputImage(FrameId fid, std::shared_ptr<ImageBuffer> input) {
  std::lock_guard<std::mutex> lock(_map_mutex);
  auto                        cache_entry_opt = _cache_map.AccessElement(fid);
  std::shared_ptr<CacheEntry> entry;
  if (cache_entry_opt.has_value()) {
    entry              = cache_entry_opt.value();
    entry->_input      = input;
    // Update last access time
    entry->last_access = std::chrono::steady_clock::now();
    _cache_map.RecordAccess(fid, entry);  // Update LRU position
  } else {
    entry         = std::make_shared<CacheEntry>();
    entry->_input = input;
    _cache_map.RecordAccess(fid, entry);
  }
}

auto PreviewPipelineStage::CurrentParamsHash() -> p_hash_t {
  nlohmann::json params_json;
  for (const auto& [op_type, entry] : _operators) {
    if (entry._enable) {
      params_json[std::to_string(static_cast<int>(op_type))] = entry._op->GetParams();
    }
  }
  return std::hash<std::string>{}(params_json.dump());
}

void PreviewPipelineStage::SetOperator(OperatorType op_type, nlohmann::json& param) {
  auto it = _operators.find(op_type);
  if (it == _operators.end()) {
    _operators.emplace(op_type,
                       OperatorEntry{true, OperatorFactory::Instance().Create(op_type, param)});
  } else {
    (it->second)._op->SetParams(param);
  }
  _cache_map.Flush();  // Invalidate cache on operator change
}

void PreviewPipelineStage::EnableOperator(OperatorType op_type, bool enable) {
  auto it = _operators.find(op_type);
  if (it != _operators.end()) {
    it->second._enable = enable;
  }
  _cache_map.Flush();  // Invalidate cache on operator change
}

void PreviewPipelineStage::SetNextStage(PreviewPipelineStage* next) { next_stage = next; }

auto PreviewPipelineStage::ApplyStage(FrameId fid) -> std::shared_ptr<ImageBuffer> {
  std::shared_ptr<CacheEntry> entry;
  {
    std::lock_guard<std::mutex> lock(_map_mutex);
    auto                        cache_entry_opt = _cache_map.AccessElement(fid);
    if (cache_entry_opt.has_value()) {
      entry = cache_entry_opt.value();
      // fast path: done and param match
      if (entry->_state == EntryState::Done && entry->params_hash == CurrentParamsHash()) {
        entry->last_access = std::chrono::steady_clock::now();
        return entry->_output;
      }
      // If another thread is computing ,wait on its future
      if (entry->_state == EntryState::InProgress) {
        auto future = entry->_shared_future;
        lock.~lock_guard();  // we release the lock before waiting
        return future.get();
      }
      // Otherwise, we need to recompute
      entry->_state         = EntryState::InProgress;
      // Reset promise/future
      entry->_promise       = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
      entry->_shared_future = entry->_promise->get_future().share();
    } else {
      entry = std::make_shared<CacheEntry>();
      // New entry, add to cache
      _cache_map.RecordAccess(fid, entry);
      entry->_state = EntryState::InProgress;
      // input should already have been set by previous stage via SetInputImage
    }
  }  // release _map_mutex lock

  // Compute the output
  if (!entry->_input) {
    throw std::runtime_error("Input image not set for preview pipeline stage " +
                             GetStageNameString());
  }
  ImageBuffer current_img = entry->_input->Clone();
  for (auto& op_pair : _operators) {
    if (op_pair.second._enable && op_pair.second._op) {
      current_img = op_pair.second._op->Apply(current_img);
    }
  }
  auto output_img = std::make_shared<ImageBuffer>(std::move(current_img));

  // Write back to cache entry
  {
    std::lock_guard<std::mutex> lock(_map_mutex);
    entry->_output     = output_img;
    entry->params_hash = CurrentParamsHash();
    entry->_state      = EntryState::Done;
    entry->last_access = std::chrono::steady_clock::now();
    // Update cache
    _cache_map.RecordAccess(fid, entry);
  }
  entry->_promise->set_value(output_img);  // Notify all waiting threads
  return output_img;
}

auto PreviewPipelineStage::GetStageNameString() const -> std::string {
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