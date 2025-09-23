#pragma once

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class ClarityOp : public OperatorBase<ClarityOp> {
 private:
  /**
   * @brief Offset to the clarity of the image, ranging from -100 to 100
   *
   */
  float        _clarity_offset;
  /**
   * @brief Scaled offset to the clarity, ranging from -1.0f to 1.0f
   *
   */
  float        _scale;

  /**
   * @brief An internal-use-only parameter to adjust the radius of the USM sharpening filter
   *
   */
  static float _usm_radius;

  void         CreateMidtoneMask(cv::Mat& input, cv::Mat& mask);

 public:
  static constexpr PriorityLevel     _priority_level    = 8;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Detail_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Clarity";
  static constexpr std::string_view  _script_name       = "clarity";
  ClarityOp();
  ClarityOp(float clarity_offset);
  ClarityOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab