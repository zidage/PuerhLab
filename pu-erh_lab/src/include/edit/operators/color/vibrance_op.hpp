#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class VibranceOp : public OperatorBase<VibranceOp> {
 private:
  /**
   * @brief An relative number for adjusting the tint,
   * negative toward green, positive toward
   *
   */
  float _vibrance_offset;

  auto  ComputeScale(float chroma) -> float;

 public:
  static constexpr std::string_view _canonical_name = "Vibrance";
  static constexpr std::string_view _script_name    = "vibrance";
  VibranceOp();
  VibranceOp(float vibrance_offset);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab