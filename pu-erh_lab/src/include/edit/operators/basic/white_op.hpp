#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "tone_region_op.hpp"

namespace puerhlab {
class WhiteOp : public ToneRegionOp<WhiteOp>, public OperatorBase<WhiteOp> {
 private:
  float           _offset;

  LinearToneCurve _curve;

 public:
  auto GetOutput(cv::Vec3f&) -> cv::Vec3f;
  auto GetOutput(hw::Vec<hw::ScalableTag<float>> luminance) -> hw::Vec<hw::ScalableTag<float>>;
  auto GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "WHITE";
  static constexpr std::string_view  _script_name       = "white";
  static constexpr ToneRegion        _tone_region       = ToneRegion::WHITE;

  WhiteOp()                                             = default;
  WhiteOp(float offset);
  WhiteOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab