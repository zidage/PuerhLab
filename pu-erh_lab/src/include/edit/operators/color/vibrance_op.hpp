#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class VibranceOp : public OperatorBase<VibranceOp> {
 private:
  /**
   * @brief An relative number for adjusting the vibrance (natural saturation)
   *
   */
  float _vibrance_offset;

  auto  ComputeScale(float chroma) -> float;

 public:
  static constexpr PriorityLevel     _priority_level    = 7;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Vibrance";
  static constexpr std::string_view  _script_name       = "vibrance";
  VibranceOp();
  VibranceOp(float vibrance_offset);
  VibranceOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab