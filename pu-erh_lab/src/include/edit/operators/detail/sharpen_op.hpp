#pragma once
#include "edit/operators/detail/sharpen_op.hpp"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class SharpenOp : public OperatorBase<SharpenOp> {
 private:
  float _offset    = 0.0f;
  float _scale     = 0.0f;

  float _radius    = 1.0f;
  float _threshold = 0.0f;

  void  ComputeScale();

 public:
  static constexpr std::string_view _canonical_name = "Sharpen";
  static constexpr std::string_view _script_name    = "sharpen";

  SharpenOp()                                       = default;
  SharpenOp(float offset, float radius, float threshold);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab