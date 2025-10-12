#pragma once

#include <opencv2/core/types.hpp>
#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "tone_region_op.hpp"

namespace puerhlab {
namespace hw = hwy::HWY_NAMESPACE;
struct HighlightCurveParams {
  float       control;
  float       knee_start;
  const float slope_range = 0.8f;
  float       m1;

  float       x0;
  float       x1 = 1.0f;
  float       y0;
  float       y1;

  float       m0 = 1.0f;

  float       dx;
};
class HighlightsOp : public ToneRegionOp<HighlightsOp>, public OperatorBase<HighlightsOp> {
 private:
  float                           _offset;

  HighlightCurveParams            _curve;
  hw::Vec<hw::ScalableTag<float>> _ctrl_param;

 public:
  auto GetOutput(hw::Vec<hw::ScalableTag<float>>) -> hw::Vec<hw::ScalableTag<float>>;
  auto GetOutput(cv::Vec3f&) -> cv::Vec3f;
  auto GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "HIGHLIGHTS";
  static constexpr std::string_view  _script_name       = "highlights";
  static constexpr ToneRegion        _tone_region       = ToneRegion::HIGHLIGHTS;

  HighlightsOp()                                        = default;
  HighlightsOp(float offset);
  HighlightsOp(const nlohmann::json& params);
  static void GetMask(cv::Mat& src, cv::Mat& mask);
  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        ToKernel() const -> Kernel override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab