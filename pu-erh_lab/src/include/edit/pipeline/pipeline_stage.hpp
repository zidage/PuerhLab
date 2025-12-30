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

namespace puerhlab {
// Iteration 3: Static Pipeline with compile-time operator chaining
template <typename... Ops>
struct PointChain {
  std::tuple<Ops...> _ops;

  PointChain(Ops... ops) : _ops(std::move(ops)...) {}

  template <size_t I = 0>
  inline void ApplyOps(Pixel& p, OperatorParams& params) {
    if constexpr (I < sizeof...(Ops)) {
      auto& op = std::get<I>(_ops);
      op(p, params);
      ApplyOps<I + 1>(p, params);
    }
  }

  void Execute(Tile& tile, OperatorParams& params) {
    int height = tile._height;
    int width  = tile._width;

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
  std::tuple<Stages...> _stages;

 public:
  StaticKernelStream(Stages... stages) : _stages(std::move(stages)...) {}
  template <size_t I = 0>
  inline void Dispatch(Tile& tile, OperatorParams& params) {
    if constexpr (I < sizeof...(Stages)) {
      auto& s = std::get<I>(_stages);

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
  bool                           _enable = true;
  std::shared_ptr<IOperatorBase> _op;

  bool                           operator<(const OperatorEntry& other) const {
    return _op->GetPriorityLevel() < other._op->GetPriorityLevel();
  }

 public:
  auto ExportOperatorParams() const -> nlohmann::json {
    nlohmann::json j;
    j["type"]   = _op->GetOperatorType();
    j["enable"] = _enable;
    j["params"] = _op->GetParams();
    return j;
  }

  auto ImportOperatorParams(const nlohmann::json& j) -> void {
    if (j.contains("enable")) _enable = j["enable"].get<bool>();
    if (j.contains("params")) _op->SetParams(j["params"]);
  }
};

class PipelineStage {
 private:
  std::unique_ptr<std::map<OperatorType, OperatorEntry>> _operators;
  bool                                                   _is_streamable      = true;
  bool                                                   _vec_enabled        = false;

  PipelineStage*                                         _prev_stage         = nullptr;
  PipelineStage*                                         _next_stage         = nullptr;

  std::shared_ptr<ImageBuffer>                           _input_img          = nullptr;
  std::shared_ptr<ImageBuffer>                           _output_cache       = nullptr;

  bool                                                   _enable_cache       = false;
  bool                                                   _input_cache_valid  = false;
  bool                                                   _output_cache_valid = false;
  bool                                                   _input_set          = false;

  PipelineStage*                                         _dependents         = nullptr;

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

  StaticKernelStreamType _static_kernel_stream = BuildKernelStream();

  std::unique_ptr<StaticTileScheduler<StaticKernelStreamType>> _static_tile_scheduler;

  GPUPipelineWrapper                                           _gpu_executor;
  bool                                                         _gpu_setup_done = false;

 public:
  PipelineStageName _stage;
  PipelineStage()                           = delete;
  PipelineStage(const PipelineStage& other) = delete;

  PipelineStage(PipelineStageName stage, bool enable_cache, bool is_streamable);

  auto IsStreamable() const -> bool { return _is_streamable; }

  void SetStaticTileScheduler() {
    _static_tile_scheduler = std::make_unique<StaticTileScheduler<StaticKernelStreamType>>(
        _input_img, _static_kernel_stream);
  }

  void SetGPUExecutor() {
    _gpu_executor.SetInputImage(_input_img);
    _gpu_setup_done = true;
  }

  void SetInputImage(std::shared_ptr<ImageBuffer>);

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
  auto GetAllOperators() const -> std::map<OperatorType, OperatorEntry>& { return *_operators; }
  void EnableOperator(OperatorType, bool enable);

  void SetNeighbors(PipelineStage* prev, PipelineStage* next) {
    _prev_stage = prev;
    _next_stage = next;
  }

  void ResetNeighbors() {
    _prev_stage = nullptr;
    _next_stage = nullptr;
  }

  void SetInputCacheValid(bool valid);
  void SetOutputCacheValid(bool valid);
  auto CacheValid() const -> bool {
    if (!_enable_cache) return false;
    if (!_prev_stage) return _output_cache_valid;
    return _input_cache_valid && _output_cache_valid;
  }

  auto GetOutputCache() const -> std::shared_ptr<ImageBuffer> { return _output_cache; }

  /**
   * @brief Used to track merged stages dependent on this stage
   *
   * @param dependent
   */
  void AddDependent(PipelineStage* dependent) { _dependents = dependent; }
  void ResetDependents() { _dependents = nullptr; }

  auto GetStageNameString() const -> std::string;

  auto HasInput() -> bool;

  auto ApplyStage(OperatorParams& global_params) -> std::shared_ptr<ImageBuffer>;

  auto GetStaticKernelStream() -> StaticKernelStreamType& { return _static_kernel_stream; }

  /**
   * @brief Reset this stage to initial state
   *
   */
  void ResetAll();

  /**
   * @brief Reset the cache of this stage
   *
   */
  void ResetCache();

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
};

}  // namespace puerhlab