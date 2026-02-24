//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "edit/operators/CPU_kernels/cpu_kernels.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/tile_scheduler.hpp"
#include "image/image_buffer.hpp"
#include "pipeline_gpu_wrapper.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {
// Iteration 3: Static Pipeline with compile-time operator chaining
template <typename... Ops>
struct PointChain {
  std::tuple<Ops...> ops_;

  PointChain(Ops... ops) : ops_(std::move(ops)...) {}

  template <size_t I = 0>
  inline void ApplyOps(Pixel& p, OperatorParams& params) {
    if constexpr (I < sizeof...(Ops)) {
      auto& op = std::get<I>(ops_);
      op(p, params);
      ApplyOps<I + 1>(p, params);
    }
  }

  void Execute(Tile& tile, OperatorParams& params) {
    int height = tile.height_;
    int width  = tile.width_;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Pixel& p = tile.at(y, x);
        ApplyOps(p, params);
      }
    }
  }
};

template <typename... Stages>
class StaticKernelStream {
  std::tuple<Stages...> stages_;

 public:
  StaticKernelStream(Stages... stages) : stages_(std::move(stages)...) {}
  template <size_t I = 0>
  inline void Dispatch(Tile& tile, OperatorParams& params) {
    if constexpr (I < sizeof...(Stages)) {
      auto& s = std::get<I>(stages_);

      if constexpr (std::is_base_of_v<PointOpTag, std::decay_t<decltype(s)>>) {
        s(tile, params);
      } else if constexpr (std::is_base_of_v<NeighborOpTag, std::decay_t<decltype(s)>>) {
        s(tile, params);
      } else {
        s.Execute(tile, params);
      }

      Dispatch<I + 1>(tile, params);
    }
  }
  void ProcessTile(Tile& tile, OperatorParams& params) { Dispatch(tile, params); }
};

struct OperatorEntry {
  bool                           enable_ = true;
  std::shared_ptr<IOperatorBase> op_;

  bool                           operator<(const OperatorEntry& other) const {
    return op_->GetPriorityLevel() < other.op_->GetPriorityLevel();
  }

 public:
  auto ExportOperatorParams() const -> nlohmann::json {
    nlohmann::json j;
    j["type"]   = op_->GetOperatorType();
    j["enable"] = enable_;
    j["params"] = op_->GetParams();
    return j;
  }

  auto ImportOperatorParams(const nlohmann::json& j) -> void {
    if (j.contains("enable")) enable_ = j["enable"].get<bool>();
    if (j.contains("params")) op_->SetParams(j["params"]);
  }
};

class PipelineStage {
 private:
  enum class StageRole { DescriptorOnly, CpuOperators, GpuStreamable, GpuOperators };

  std::unique_ptr<std::map<OperatorType, OperatorEntry>> operators_;
  bool                                                   is_streamable_ = true;
  bool                                                   vec_enabled_   = false;

  StageRole                                              stage_role_    = StageRole::DescriptorOnly;

  PipelineStage*                                         prev_stage_    = nullptr;
  PipelineStage*                                         next_stage_    = nullptr;

  std::shared_ptr<ImageBuffer>                           input_img_     = nullptr;
  std::shared_ptr<ImageBuffer>                           output_cache_  = nullptr;

  bool                                                   enable_cache_  = false;
  bool                                                   input_cache_valid_  = false;
  bool                                                   output_cache_valid_ = false;
  bool                                                   input_set_          = false;

  PipelineStage*                                         dependents_         = nullptr;

  static constexpr auto                                  BuildKernelStream   = []() {
    auto op_to_working = OCIO_ACES_Transform_Op_Kernel();
    auto op_exp        = ExposureOpKernel();
    auto op_contrast   = ContrastOpKernel();
    auto op_black      = BlackOpKernel();
    auto op_white      = WhiteOpKernel();
    auto op_shadow     = ShadowsOpKernel();
    auto op_highlight  = HighlightsOpKernel();

    auto op_curve      = CurveOpKernel();

    auto op_tint       = TintOpKernel();
    auto op_saturation = SaturationOpKernel();
    auto op_vibrance   = VibranceOpKernel();

    auto color_wheel   = ColorWheelOpKernel();

    auto op_lmt        = OCIO_LMT_Transform_Op_Kernel();

    auto op_clarity    = ClarityOpKernel();
    auto op_sharpen    = SharpenOpKernel();

    auto to_output     = OCIO_ACES_Transform_Op_Kernel();

    return StaticKernelStream(
        PointChain(op_to_working, op_exp, op_contrast, op_black, op_white, op_shadow, op_highlight,
                                                      op_curve, op_tint, op_saturation, op_vibrance, color_wheel, op_lmt, to_output),
        op_clarity, op_sharpen);
  };

  using StaticKernelStreamType                 = decltype(BuildKernelStream());

  StaticKernelStreamType static_kernel_stream_ = BuildKernelStream();

  std::unique_ptr<StaticTileScheduler<StaticKernelStreamType>> static_tile_scheduler_;

  GPUPipelineWrapper                                           gpu_executor_;
  bool                                                         gpu_setup_done_ = false;

 public:
  enum class RuntimeResetMode {
    InvalidateCache,
    ClearIntermediateBuffers,
    ReleaseGpuResources,
    ClearIntermediateBuffersAndGpu,
  };

  PipelineStageName stage_;
  PipelineStage()                           = delete;
  PipelineStage(const PipelineStage& other) = delete;

  PipelineStage(PipelineStageName stage, bool enable_cache, bool is_streamable);

  auto IsStreamable() const -> bool { return is_streamable_; }

  void SetStaticTileScheduler() {
    static_tile_scheduler_ = std::make_unique<StaticTileScheduler<StaticKernelStreamType>>(
        input_img_, static_kernel_stream_);
  }

  void SetGPUExecutor() {
    gpu_executor_.SetInputImage(input_img_);
    gpu_setup_done_ = true;
  }

  void SetFrameSink(IFrameSink* frame_sink) { gpu_executor_.SetFrameSink(frame_sink); }

  void SetInputImage(std::shared_ptr<ImageBuffer>);

  void SetForceCPUOutput(bool force) { force_cpu_output_ = force; }

  // Unified runtime resource reset entrypoint.
  // Does not modify operator configuration.
  void ResetRuntimeResources(RuntimeResetMode mode);

  /**
   * @brief Set the parameters for an operator with the given type in this stage.
   *
   * @param op_type
   * @param param
   * @return int 1 if a new operator is created and added (if this stage was merged into the
   * execution stage, should call SetExecutionStages() to rebuild the kernel stream), 0 if an
   * existing operator is updated.
   */
  void SetOperator(OperatorType, nlohmann::json param);

  void SetOperator(OperatorType, nlohmann::json param, OperatorParams& global_params);

  auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*>;
  auto GetAllOperators() const -> std::map<OperatorType, OperatorEntry>& { return *operators_; }
  void EnableOperator(OperatorType, bool enable);
  void EnableOperator(OperatorType, bool enable, OperatorParams& global_params);

  void SetNeighbors(PipelineStage* prev, PipelineStage* next) {
    prev_stage_ = prev;
    next_stage_ = next;
  }

  void ResetNeighbors() {
    prev_stage_ = nullptr;
    next_stage_ = nullptr;
  }

  void SetInputCacheValid(bool valid);
  void SetOutputCacheValid(bool valid);
  auto CacheValid() const -> bool {
    if (!enable_cache_) return false;
    if (!prev_stage_) return output_cache_valid_;
    return input_cache_valid_ && output_cache_valid_;
  }

  auto GetOutputCache() const -> std::shared_ptr<ImageBuffer> { return output_cache_; }

  /**
   * @brief Used to track merged stages dependent on this stage
   *
   * @param dependent
   */
  void AddDependent(PipelineStage* dependent) { dependents_ = dependent; }
  void ResetDependents() { dependents_ = nullptr; }

  auto GetStageNameString() const -> std::string;

  auto HasInput() -> bool;

  auto ApplyStage(OperatorParams& global_params) -> std::shared_ptr<ImageBuffer>;

  auto GetStaticKernelStream() -> StaticKernelStreamType& { return static_kernel_stream_; }

  void SetEnableCache(bool enable) {
    if (enable_cache_ == enable) return;
    enable_cache_ = enable;
    ResetRuntimeResources(RuntimeResetMode::InvalidateCache);
  }

  /**
   * @brief Reset this stage to initial state
   *
   */
  void ResetAll();

  /**
   * @brief Export the parameters of this stage and its operators to JSON (serialize)
   *
   * @return nlohmann::json
   */
  auto ExportStageParams() const -> nlohmann::json;

  /**
   * @brief Import the parameters of this stage and its operators from JSON (deserialize)
   *
   * @param j
   */
  void ImportStageParams(const nlohmann::json& j);

  void ImportStageParams(const nlohmann::json& j, OperatorParams& global_params);

 private:
  static StageRole             DetermineStageRole(PipelineStageName stage, bool is_streamable);

  bool                         HasEnabledOperator() const;

  std::shared_ptr<ImageBuffer> ApplyDescriptorOnly();
  std::shared_ptr<ImageBuffer> ApplyCpuOperators();
  std::shared_ptr<ImageBuffer> ApplyGpuOperators(OperatorParams& global_params);
  std::shared_ptr<ImageBuffer> ApplyGpuStream(OperatorParams& global_params);

  bool                         force_cpu_output_ = false;
};

}  // namespace puerhlab
