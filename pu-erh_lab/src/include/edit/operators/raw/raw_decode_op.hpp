#pragma once

#include <memory>

#include "decoders/processor/raw_processor.hpp"
#include "edit/operators/op_base.hpp"

namespace puerhlab {

enum class RawProcessBackend { PUERH, LIBRAW };

class RawDecodeOp : public OperatorBase<RawDecodeOp> {
 public:
  static constexpr PriorityLevel     _priority_level    = 0;
  static constexpr std::string_view  _canonical_name    = "RawDecode";
  static constexpr std::string_view  _script_name       = "raw_decode";
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Image_Loading;

  RawParams                          _params;
  RawProcessBackend                  _backend;

  RawDecodeOp() = delete;

  RawDecodeOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override {
    throw std::runtime_error("RawDecodeOp does not support kernel processing.");
  }
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};

}  // namespace puerhlab