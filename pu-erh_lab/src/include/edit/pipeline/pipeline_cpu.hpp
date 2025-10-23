#pragma once

#include <memory>

#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline_stage.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "pipeline_stage.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
class CPUPipelineExecutor : public PipelineExecutor {
 private:
  bool                                                                        _enable_cache = true;
  std::array<PipelineStage, static_cast<int>(PipelineStageName::Stage_Count)> _stages;

  KernelStream                                                                _kernel_stream;

  bool                                                                        _is_thumbnail = false;

  nlohmann::json                                                              _thumbnail_params;

  static constexpr PipelineBackend            _backend = PipelineBackend::CPU;

  std::vector<PipelineStage*>                 _exec_stages;
  std::vector<std::unique_ptr<PipelineStage>> _merged_stages;

  void                                        ResetStages();

 public:
  CPUPipelineExecutor();
  CPUPipelineExecutor(bool enable_cache);

  void SetEnableCache(bool enable_cache);
  auto GetBackend() -> PipelineBackend override;

  auto GetStage(PipelineStageName stage) -> PipelineStage& override;
  auto Apply(std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<ImageBuffer> override;

  void SetPreviewMode(bool is_preview);
  void SetExecutionStages();
  void ResetExecutionStages();
};
};  // namespace puerhlab