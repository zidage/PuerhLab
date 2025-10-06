#pragma once

#include <json.hpp>
#include <string>
#include <string_view>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "tone_region_op.hpp"

namespace puerhlab {
namespace hw = hwy::HWY_NAMESPACE;
class BlackOp : public ToneRegionOp<BlackOp>, public OperatorBase<BlackOp> {
 private:
  float           _offset;

  float           _y_intercept;
  float           _slope;               // slope of the tone curve
  LinearToneCurve _curve;

 public:
  auto GetOutput(cv::Vec3f&) -> cv::Vec3f;
  auto GetOutput(hw::Vec<hw::ScalableTag<float>> luminance) -> hw::Vec<hw::ScalableTag<float>>;
  auto GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "BLACK";
  static constexpr std::string_view  _script_name       = "black";
  static constexpr ToneRegion        _tone_region       = ToneRegion::BLACK;

  BlackOp()                                             = default;
  BlackOp(float offset);
  BlackOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        ToKernel() const -> Kernel override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab