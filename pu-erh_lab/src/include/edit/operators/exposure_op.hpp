#pragma once

#include <string>
#include <string_view>

#include "op_base.hpp"

namespace puerhlab {
class ExposureOp : public OperatorBase<ExposureOp> {
 private:
  /**
   * @brief An EV offset applied to the target image.
   * Positive to increase the brightness, negative to darken.
   *
   */
  float                             _exposure_offset;

  static constexpr std::string_view _canonical_name = "Exposure";
  static constexpr std::string_view _script_name    = "exposure";

 public:
  ExposureOp();
  ExposureOp(float exposure_offset);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab