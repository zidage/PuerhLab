#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class SaturationOp : public OperatorBase<SaturationOp> {
 private:
  /**
   * @brief An relative number for adjusting the saturation from -100 to 100
   *
   */
  float _saturation_offset;

  /**
   * @brief The absolute value for the saturation adjustment from -1.0f to 1.0f
   *
   */
  float _scale;

  void  ComputeScale();

 public:
  static constexpr PriorityLevel     _priority_level    = 6;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Saturation";
  static constexpr std::string_view  _script_name       = "saturation";
  SaturationOp();
  SaturationOp(float saturation_offset);
  SaturationOp(const nlohmann::json& params);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab