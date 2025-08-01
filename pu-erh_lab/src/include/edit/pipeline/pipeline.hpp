#pragma once

#include "image/image_buffer.hpp"
#include "pipeline_utils.hpp"

namespace puerhlab {
class Pipeline {
 public:
  virtual auto GetStage(PipelineStageName stage) -> PipelineStage&;
  virtual auto Apply(ImageBuffer& input) -> ImageBuffer;
};
}  // namespace puerhlab
