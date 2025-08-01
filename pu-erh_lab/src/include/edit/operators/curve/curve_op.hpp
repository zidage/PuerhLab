#pragma once

#include <opencv2/core/types.hpp>
#include <vector>

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct CurveOpRegister {
  CurveOpRegister();
};

class CurveOp : public OperatorBase<CurveOp> {
 private:
  std::vector<cv::Point2f> _ctrl_pts;
  std::vector<float>       _h;
  std::vector<float>       _m;

  void                     ComputeTagents();
  auto                     EvaluateCurve(float x) const -> float;

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Curve";
  static constexpr std::string_view  _script_name       = "curve";
  CurveOp()                                             = delete;
  CurveOp(const std::vector<cv::Point2f>& control_points);
  CurveOp(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  void SetCtrlPts(const std::vector<cv::Point2f>& control_points);
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
};  // namespace puerhlab
