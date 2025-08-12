#pragma once

#include <opencv2/core.hpp>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

class CVCvtColorOp : public OperatorBase<CVCvtColorOp> {
 private:
  static void CalculateOutput(cv::Vec3f& color, float adapted_lum = 1.0f);

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "CV CvtColor";
  static constexpr std::string_view  _script_name       = "CV_CvtColor";

  CVCvtColorOp()                                        = default;
  CVCvtColorOp(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab