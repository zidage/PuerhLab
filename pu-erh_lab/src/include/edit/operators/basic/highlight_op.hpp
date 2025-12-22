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
  static constexpr OperatorType      _operator_type     = OperatorType::HIGHLIGHTS;

  HighlightsOp()                                        = default;
  HighlightsOp(float offset);
  HighlightsOp(const nlohmann::json& params);
  static void GetMask(cv::Mat& src, cv::Mat& mask);
  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        ToKernel() const -> Kernel override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;

  inline void operator()(Pixel& p, OperatorParams& params) const {
    float L    = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    float outL = L;
    if (L <= params.highlights_k) {
      // below knee_start: identity
      outL = L;
    } else if (L < 1.0f) {
      // inside the Hermite segment: parameterize t in [0,1]
      float t   = (L - params.highlights_k) / params.highlights_dx;
      // Hermite interpolation:
      float H00 = 2 * t * t * t - 3 * t * t + 1;
      float H10 = t * t * t - 2 * t * t + t;
      float H01 = -2 * t * t * t + 3 * t * t;
      float H11 = t * t * t - t * t;
      // note: tangents in Hermite are (dx * m0) and (dx * m1)
      outL      = H00 * params.highlights_k + H10 * (params.highlights_dx * params.highlights_m0) + H01 * 1.0f + H11 * (params.highlights_dx * params.highlights_m1);
    } else {
      // L >= whitepoint: linear extrapolate using slope m1
      outL = 1.0f + (L - 1.0f) * params.highlights_m1;
    }

    // avoid negative or NaN
    if (!std::isfinite(outL)) outL = L;
    // Preserve hue/chroma by scaling RGB by ratio outL/L (guard L==0)
    float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
    p.r *= scale;
    p.g *= scale;
    p.b *= scale;
  }
};
}  // namespace puerhlab