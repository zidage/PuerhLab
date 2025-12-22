#pragma once

#include <opencv2/core/types.hpp>
#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"

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
class HighlightsOp : public OperatorBase<HighlightsOp> {
 private:
  float                _offset;

  HighlightCurveParams _curve;

 public:
  auto                               GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "HIGHLIGHTS";
  static constexpr std::string_view  _script_name       = "highlights";
  static constexpr OperatorType      _operator_type     = OperatorType::HIGHLIGHTS;

  HighlightsOp()                                        = default;
  HighlightsOp(float offset);
  HighlightsOp(const nlohmann::json& params);
  static void GetMask(cv::Mat& src, cv::Mat& mask);
  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;

  void        SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab