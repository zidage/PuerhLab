#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class TintOp : public OperatorBase<TintOp> {
 private:
  /**
   * @brief An relative number for adjusting the tint,
   * negative toward green, positive toward magneta
   * Range from -100 to 100
   */
  float _tint_offset;

  /**
   * @brief An absolute number for adjusting the tint,
   * negative toward green, positive toward magneta
   * Range from -1.0f to 1.0f
   */
  float _scale;

 public:
  static constexpr std::string_view _canonical_name = "Tint";
  static constexpr std::string_view _script_name    = "tint";
  TintOp();
  TintOp(float tint_offset);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab