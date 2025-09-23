#pragma once

#include <memory>

#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline_utils.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "utils/cache/lru_cache.hpp"


namespace puerhlab {
class CPUPipelineExecutor : public PipelineExecutor {
 private:
  std::unique_ptr<std::array<PipelineStage, static_cast<int>(PipelineStageName::Stage_Count)>>
                                   _stages;

  bool                             _is_thumbnail = false;
  bool                             _enable_cache = true;
  nlohmann::json                   _thumbnail_params;

  static constexpr PipelineBackend _backend = PipelineBackend::CPU;

  void                             ResetStages();

 public:
  CPUPipelineExecutor();
  CPUPipelineExecutor(bool enable_cache);

  void SetEnableCache(bool enable_cache);
  auto GetBackend() -> PipelineBackend override;

  auto GetStage(PipelineStageName stage) -> PipelineStage& override;
  auto Apply(std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<ImageBuffer> override;

  void SetThumbnailMode(bool is_thumbnail);
};
};  // namespace puerhlab