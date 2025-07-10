#pragma once

#include <opencv2/core/types.hpp>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class ColorWheelOp : OperatorBase<ColorWheelOp> {
 public:
  struct WheelControl {
    // x for hue (0->360.0f), y for saturation (0->1)
    cv::Point3f color_offset{0.0f, 0.0f, 0.0f};
    float       luminance_offset{0.0f};
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(WheelControl, color_offset.x, color_offset.y, color_offset.z,
                                   luminance_offset)
  };

 private:
  WheelControl _lift;
  WheelControl _gamma;
  WheelControl _gain;

  float        _lift_crossover;
  float        _gain_crossover;

 public:
  static constexpr std::string_view _canonical_name = "Color Wheel";
  static constexpr std::string_view _script_name    = "color_wheel";

  ColorWheelOp();

  auto Apply(ImageBuffer& input) -> ImageBuffer override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
};
};  // namespace puerhlab

// NLOHMANN_DEFINE_TYPE_INTRUSIVE
namespace cv {
inline void to_json(nlohmann::json& j, const Point3f& p) {
  j = {{"x", p.x}, {"y", p.y}, {"z", p.z}};
}
inline void from_json(const nlohmann::json& j, Point3f& p) {
  j.at("x").get_to(p.x);
  j.at("y").get_to(p.y);
  j.at("z").get_to(p.z);
}
}  // namespace cv