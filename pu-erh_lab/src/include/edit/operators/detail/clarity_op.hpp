#pragma once

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class ClarityOp : public OperatorBase<ClarityOp> {
 private:
  float        _clarity_offset;
  float        _scale;

  static float _usm_radius;
  static float _blur_sigma;

  void         CreateMidtoneMask(cv::Mat& input, cv::Mat& mask);

 public:
  static constexpr std::string_view _canonical_name = "Clarity";
  static constexpr std::string_view _script_name    = "clarity";
  ClarityOp();
  ClarityOp(float clarity_offset);

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab