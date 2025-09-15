#pragma once

#include <cstdint>
#include <future>
#include <memory>

#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline_utils.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "preview_pipeline_utils.hpp"
#include "utils/cache/lru_cache.hpp"

namespace puerhlab {
class CPU_PreviewPipelineExecutor : public PipelineExecutor {
 private:
  std::array<PreviewPipelineStage, static_cast<int>(PipelineStageName::Stage_Count)> _stages;

  // Preview pipeline generate thumbnail defaultly
  nlohmann::json                   _thumbnail_params;

  static constexpr PipelineBackend _backend = PipelineBackend::CPU;

  auto ExecuteStage(size_t idx, uint64_t frame_id, std::shared_ptr<ImageBuffer> input)
      -> std::shared_future<std::shared_ptr<ImageBuffer>>;

 public:
  CPU_PreviewPipelineExecutor();

  auto GetBackend() -> PipelineBackend override;

  auto GetStage(PipelineStageName stage) -> PipelineStage& override;

  auto GetPreviewStage(PipelineStageName stage) -> PreviewPipelineStage&;
  auto Apply(ImageBuffer& input) -> ImageBuffer override;

  auto Apply(FrameId id, ImageBuffer& input) -> std::shared_future<ImageBuffer>;

  void SetThumbnailMode(bool is_thumbnail);
};
};  // namespace puerhlab