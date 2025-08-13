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
  float                                 _offset;

  hw::Vec<hw::ScalableTag<float>>       _scale;
  const hw::ScalableTag<float>          d;
  const hw::Vec<hw::ScalableTag<float>> v_zero          = hw::Set(d, 0.0f);
  const hw::Vec<hw::ScalableTag<float>> v_one           = hw::Set(d, 1.0f);
  const hw::Vec<hw::ScalableTag<float>> v_inv_threshold = hw::Set(d, 1.0f / 0.5f);
  const hw::Vec<hw::ScalableTag<float>> v_epsilon       = hw::Set(d, 1e-7f);

  const hw::Vec<hw::ScalableTag<float>> _pivot          = hw::Set(d, 0.05f);

  hw::Vec<hw::ScalableTag<float>>       v_gamma;
  float                                 _gamma;
  float                                 _inv_threshold = 1.0f / 0.5f;

  hw::Vec<hw::ScalableTag<float>>       _min           = hw::Set(hw::ScalableTag<float>(), 0.0f);
  hw::Vec<hw::ScalableTag<float>>       _max           = hw::Set(hw::ScalableTag<float>(), 100.0f);

 public:
  auto GetOutput(hw::Vec<hw::ScalableTag<float>>) -> hw::Vec<hw::ScalableTag<float>>;
  auto GetOutput(float luminance) -> float;
  auto GetScale() -> float;
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