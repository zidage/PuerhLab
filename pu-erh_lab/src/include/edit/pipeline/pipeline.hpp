#pragma once

#include "image/image_buffer.hpp"
#include "pipeline_utils.hpp"

namespace puerhlab {
class PipelineExecutor {
 public:
  virtual auto GetStage(PipelineStageName stage) -> PipelineStage& = 0;
  virtual auto Apply(ImageBuffer& input) -> ImageBuffer            = 0;
};
}  // namespace puerhlab
