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
  static constexpr std::string_view _canonical_name = "Saturation";
  static constexpr std::string_view _script_name    = "saturation";
  SaturationOp();
  SaturationOp(float saturation_offset);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab