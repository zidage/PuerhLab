#pragma once

#include "image/image_buffer.hpp"
#include "pipeline_utils.hpp"

namespace puerhlab {
enum class PipelineBackend { CPU, CUDA, OpenCL };
class PipelineExecutor {
 public:
  virtual auto GetStage(PipelineStageName stage) -> PipelineStage& = 0;
  virtual auto SetOperator(PipelineStageName stage, OperatorType type, nlohmann::json& param)
      -> bool                                           = 0;
  virtual auto Apply(ImageBuffer& input) -> ImageBuffer = 0;
  virtual auto GetBackend() -> PipelineBackend          = 0;
};
}  // namespace puerhlab
