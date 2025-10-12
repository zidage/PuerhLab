#pragma once

#include <hwy/highway.h>

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ExposureOp : public OperatorBase<ExposureOp> {
 private:
  /**
   * @brief An EV offset applied to the target image.
   * Positive to increase the brightness, negative to darken.
   *
   */
  float                           _exposure_offset;

  /**
   * @brief The actual luminance offset derived from the EV
   * dL = 2^E
   *
   */
  float                           _scale;

  float                           _gamma;


 public:
  static constexpr PriorityLevel     _priority_level    = 0;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Exposure";
  static constexpr std::string_view  _script_name       = "exposure";
  ExposureOp();
  ExposureOp(float exposure_offset);
  ExposureOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override;
  auto ToKernel_Vec() const -> Kernel; // not implemented yet
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab