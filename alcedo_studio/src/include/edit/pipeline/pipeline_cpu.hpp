//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>
#include <mutex>
#include <optional>

#include "edit/operators/op_base.hpp"
#include "edit/pipeline/pipeline_stage.hpp"
#include "image/image_buffer.hpp"
#include "pipeline.hpp"
#include "pipeline_stage.hpp"
#include "type/type.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace alcedo {
class CPUPipelineExecutor : public PipelineExecutor {
 private:
  sl_element_id_t                                                             bound_file_id_ = 0;
  bool                                                                        enable_cache_  = true;
  std::array<PipelineStage, static_cast<int>(PipelineStageName::Stage_Count)> stages_;

  // The executor state (render params, stage cache, decode mode) is mutable and not thread-safe.
  // Serialize concurrent scheduler tasks that target the same executor instance.
  std::mutex                                                                  render_lock_;

  OperatorParams                                                              global_params_;

  bool                                                                        is_thumbnail_ = false;

  bool                             force_cpu_output_                                        = false;
  DecodeRes                        decode_res_    = DecodeRes::FULL;

  nlohmann::json                   render_params_ = {};

  static constexpr PipelineBackend backend_       = PipelineBackend::CPU;

  std::vector<PipelineStage*>      exec_stages_;
  std::unique_ptr<PipelineStage>   merged_stages_;
  IFrameSink*                      frame_sink_      = nullptr;

  void                             ResetStages();

  void                             ResetExecutionStagesCache();

  void                             SetTemplateParams();

 public:
  CPUPipelineExecutor();
  CPUPipelineExecutor(bool enable_cache);

  void SetBoundFile(sl_element_id_t file_id) override { bound_file_id_ = file_id; }
  auto GetBoundFile() const -> sl_element_id_t override { return bound_file_id_; }

  void SetEnableCache(bool enable_cache);
  auto GetBackend() -> PipelineBackend override;

  void SetForceCPUOutput(bool force) override { force_cpu_output_ = force; }

  auto GetRenderLock() -> std::mutex& { return render_lock_; }

  auto GetStage(PipelineStageName stage) -> PipelineStage& override;
  auto Apply(std::shared_ptr<ImageBuffer> input) -> std::shared_ptr<ImageBuffer> override;

  void SetPreviewMode(bool is_preview);

  void SetExecutionStages();
  void SetExecutionStages(IFrameSink* frame_sink);
  void ResetExecutionStages();

  auto GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion>;
  void SetNextFramePresentationMode(FramePresentationMode mode) const;

  auto GetGlobalParams() -> OperatorParams& override { return global_params_; }

  /**
   * @brief Serialize the pipeline parameters to JSON
   *
   * @return nlohmann::json
   */
  auto ExportPipelineParams() const -> nlohmann::json override;
  /**
   * @brief Set the pipeline parameters from JSON. It will reset all stages and operators, as well
   * as cache. After importing, you need to call SetExecutionStages() to rebuild the execution
   * stages.
   *
   * @param j
   */
  void ImportPipelineParams(const nlohmann::json& j) override;

  void SetRenderRegion(int x, int y, float scale_factor_x,
                       float scale_factor_y = -1.0f) override;
  void SetRenderRes(bool full_res, int max_side_length = 2048) override;
  void SetResizeDownsampleAlgorithm(ResizeDownsampleAlgorithm algorithm) override;
  void SetDecodeRes(DecodeRes res);

  void RegisterAllOperators();

  void InitDefaultPipeline();

  /**
   * @brief Inject pre-extracted raw metadata into the pipeline.
   *        Populates global params so that downstream operators (ColorTemp, LensCalib)
   *        can resolve eagerly, and sets the pre-populated context on RawDecodeOp
   *        so it skips re-extraction at decode time.
   */
  void InjectRawMetadata(const RawRuntimeColorContext& ctx);

  /**
   * @brief Clear all intermediate image buffers from all stages.
   *        Call this after pipeline execution when you want to release memory
   *        while keeping the pipeline configuration intact.
   */
  void ClearAllIntermediateBuffers();

  /**
   * @brief Release transient merged-stage preview scratch buffers while keeping
   *        the compiled GPU pipeline and LUT state intact.
   *        Use this when a full-resolution preview/export frame returns to the
   *        FAST_PREVIEW baseline and the large scratch high-water mark should
   *        not stay pinned in VRAM.
   */
  void ReleasePreviewGpuScratch();

  /**
   * @brief Release persistent GPU allocations held by execution stages.
   *        Useful for batch export to avoid holding large VRAM allocations
   *        across many cached pipelines.
   */
  void ReleaseAllGPUResources();

  [[nodiscard]] auto DebugGetMergedStageScratchBytes() const -> size_t;
};
};  // namespace alcedo
