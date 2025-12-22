#pragma once

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "pipeline_stage.hpp"

namespace puerhlab {
enum class PipelineBackend { CPU, CUDA, OpenCL };
class PipelineExecutor {
 public:
  virtual auto GetStage(PipelineStageName stage) -> PipelineStage&                       = 0;
  virtual auto Apply(std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<ImageBuffer> = 0;
  virtual auto GetBackend() -> PipelineBackend                                           = 0;
  virtual auto ExportPipelineParams() const -> nlohmann::json                            = 0;
  virtual void ImportPipelineParams(const nlohmann::json& j)                             = 0;

  virtual auto GetGlobalParams() -> OperatorParams&                                      = 0;
};

}  // namespace puerhlab
