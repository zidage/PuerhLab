#include "edit/operators/op_base.hpp"

namespace puerhlab {
class SaturationOp : public OperatorBase<SaturationOp> {
 private:
  /**
   * @brief An relative number for adjusting the tint,
   * negative toward green, positive toward
   *
   */
  float _saturation_offset;

  float _scale;

  void  ComputeScale();

 public:
  static constexpr std::string_view _canonical_name = "SaturationOp";
  static constexpr std::string_view _script_name    = "saturation";
  SaturationOp();
  SaturationOp(float saturation_offset);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab