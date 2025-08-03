#pragma once

#include "edit/pipeline/pipeline_utils.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
class CPUPipeline : public Pipeline {
 private:
  std::array<PipelineStage, static_cast<int>(PipelineStageName::Stage_Count)> _stages;

  // TODO: Caching (NOT IMPLEMENTED)
  LRUCache<PipelineStageName, ImageBuffer>                                    _image_buffer;

 public:
  CPUPipeline();
  CPUPipeline(const CPUPipeline& other) = default;
  auto GetStage(PipelineStageName stage) -> PipelineStage& override;
  auto Apply(ImageBuffer& input) -> ImageBuffer override;
};
};  // namespace puerhlab