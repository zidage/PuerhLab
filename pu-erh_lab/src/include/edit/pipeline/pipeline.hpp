#pragma once

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
};

// Iteration 3: Static Pipeline with compile-time operator chaining
template <typename... Ops>
struct PointChain {
  std::tuple<Ops...> _ops;

  PointChain(Ops... ops) : _ops(std::move(ops)...) {}

  void Execute(Tile& tile, OperatorParams& params) {
    int height = tile._height;
    int width  = tile._width;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Pixel& p = tile.at(y, x);
        std::apply([&p, &params](auto&... op) { (op(p, params), ...); }, _ops);
      }
    }
  }
};

template <typename... Stages>
class StaticPipeline {
  std::tuple<Stages...> _stages;

 public:
  StaticPipeline(Stages... stages) : _stages(std::move(stages)...) {}

  void ProcessTile(Tile& tile, OperatorParams& params) {
    std::apply([&](auto&... stage) {
      auto dispatch = [&](auto& s) {
        if constexpr (std::is_base_of_v<PointOpTag, std::decay_t<decltype(s)>>) {
          
        } else if constexpr (std::is_base_of_v<NeighborOpTag, std::decay_t<decltype(s)>>) {
          s(tile, params);
        } else {
          s.Execute(tile, params);
        }
      };
    }, _stages);
  }
};
}  // namespace puerhlab
