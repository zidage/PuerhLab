#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "tone_region_op.hpp"
#include "utils/simd/simple_simd.hpp"

namespace puerhlab {
class WhiteOp : public ToneRegionOp<WhiteOp>, public OperatorBase<WhiteOp> {
 private:
  float           _offset;

  float           _y_intercept;
  float           _black_point;
  float           _slope;

  LinearToneCurve _curve;

  simple_simd::f32x4 _y_intercept_vec;
  simple_simd::f32x4 _black_point_vec;
  simple_simd::f32x4 _slope_vec;

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
  auto        ToKernel() const -> Kernel override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab