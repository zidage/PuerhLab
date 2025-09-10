#pragma once

#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline_utils.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
class CPUPipelineExecutor : public PipelineExecutor {
 private:
  std::array<PipelineStage, static_cast<int>(PipelineStageName::Stage_Count)> _stages;

  bool                                                                        _is_thumbnail = false;
  nlohmann::json                                                              _thumbnail_params;

  // TODO: Caching (NOT IMPLEMENTED)
  LRUCache<PipelineStageName, ImageBuffer>                                    _image_buffer;

 public:
  CPUPipelineExecutor();
  auto GetStage(PipelineStageName stage) -> PipelineStage& override;
  auto Apply(ImageBuffer& input) -> ImageBuffer override;

  void SetThumbnailMode(bool is_thumbnail) {
    _is_thumbnail     = is_thumbnail;
    _thumbnail_params = {};  // TODO: Use default params for now
    if (!_is_thumbnail) {
      // Disable resizing in image loading stage
      _stages[static_cast<int>(PipelineStageName::Image_Loading)].EnableOperator(
          OperatorType::RESIZE,
          false);  // If RESIZE operator not exist, this function will do nothing
      return;
    }
    _stages[static_cast<int>(PipelineStageName::Image_Loading)].SetOperator(OperatorType::RESIZE,
                                                                            _thumbnail_params);
  }
};
};  // namespace puerhlab