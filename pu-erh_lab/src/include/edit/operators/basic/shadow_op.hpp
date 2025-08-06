#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "tone_region_op.hpp"

namespace puerhlab {

class ShadowsOp : public ToneRegionOp<ShadowsOp>, public OperatorBase<ShadowsOp> {
 private:
  float _offset;

 public:
  static auto                        GetOutput(float luminance, float adj) -> float;
  auto                               GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Shadows";
  static constexpr std::string_view  _script_name       = "shadows";
  static constexpr ToneRegion        _tone_region       = ToneRegion::SHADOWS;

  ShadowsOp()                                           = default;
  ShadowsOp(float offset);
  ShadowsOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  auto        Apply(ImageBuffer& input) -> ImageBuffer override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab