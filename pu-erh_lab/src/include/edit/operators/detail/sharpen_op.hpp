#pragma once
#include <vector>

#include "edit/operators/detail/sharpen_op.hpp"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class SharpenOp : public OperatorBase<SharpenOp> {
 private:
  /**
   * @brief Offset to the sharpness of the image, ranging from 0 to 100
   *
   */
  float              _offset    = 0.0f;
  /**
   * @brief Scaled offset to the sharpness of the image, ranging from 0 to 1.0f
   *
   */
  float              _scale     = 0.0f;

  /**
   * @brief The USM radius
   *
   */
  float              _radius    = 1.0f;
  /**
   * @brief A threshold limiting the sharpening effect, like the "Mask" option in ACR's sharpening
   * module
   *
   */
  float              _threshold = 0.0f;

  std::vector<float> _kernel;

  void               ComputeScale();

 public:
  static constexpr PriorityLevel     _priority_level    = 8;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Detail_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Sharpen";
  static constexpr std::string_view  _script_name       = "sharpen";

  SharpenOp()                                           = default;
  SharpenOp(float offset, float radius, float threshold);
  SharpenOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto ToKernel() const -> Kernel override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab