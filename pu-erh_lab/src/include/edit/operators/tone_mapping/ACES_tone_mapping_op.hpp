#include <opencv2/core.hpp>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

class ACESToneMappingOp : public OperatorBase<ACESToneMappingOp> {
 private:
  static void CalculateOutput(cv::Vec3f& color, float adapted_lum = 1.0f);

 public:
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Output_Transform;
  static constexpr std::string_view  _canonical_name    = "ACES ToneMapping";
  static constexpr std::string_view  _script_name       = "ACES_tone_mapping";

  ACESToneMappingOp()                                   = default;
  ACESToneMappingOp(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab