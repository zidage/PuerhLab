#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

class ContrastOp : public OperatorBase<ContrastOp> {
 private:
  /**
   * @brief A relative number for adjusting the image
   *
   */
  float _contrast_offset;
  /**
   * @brief An absolute number to represent the contrast after adjustment
   * Usually, it is computed through dividing 100.0f from the offset
   *
   */
  float _scale;

 public:
  static constexpr PriorityLevel     _priority_level    = 3;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Contrast";
  static constexpr std::string_view  _script_name       = "contrast";
  ContrastOp();
  ContrastOp(float contrast_offset);
  ContrastOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab