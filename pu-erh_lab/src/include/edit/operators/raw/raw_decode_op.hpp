#pragma once

#include "decoders/processor/raw_processor.hpp"
#include "edit/operators/op_base.hpp"

namespace puerhlab {

class RawDecodeOp : public OperatorBase<RawDecodeOp> {
 public:
  static constexpr PriorityLevel     _priority_level    = 0;
  static constexpr std::string_view  _canonical_name    = "RawDecode";
  static constexpr std::string_view  _script_name       = "raw_decode";
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Raw_Decoding;

  RawParams                          _params;
  OpenCVRawProcessor                 _processor;

  RawDecodeOp() = delete;

  RawDecodeOp(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};

}  // namespace puerhlab