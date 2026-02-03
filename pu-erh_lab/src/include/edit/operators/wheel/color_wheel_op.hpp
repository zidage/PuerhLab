//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <opencv2/core/types.hpp>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class ColorWheelOp : public OperatorBase<ColorWheelOp> {
 public:
  struct WheelControl {
    // x for hue (0->360.0f), y for saturation (0->1)
    cv::Point3f color_offset_{0.0f, 0.0f, 0.0f};
    float       luminance_offset_{0.0f};
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(WheelControl, color_offset_.x, color_offset_.y, color_offset_.z,
                                   luminance_offset_)
  };

 private:
  WheelControl lift_;
  WheelControl gamma_;
  WheelControl gain_;

  float        lift_crossover_;
  float        gain_crossover_;

 public:
  static constexpr PriorityLevel     priority_level_    = 5;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Color Wheel";
  static constexpr std::string_view  script_name_       = "color_wheel";
  static constexpr OperatorType      operator_type_     = OperatorType::COLOR_WHEEL;
  ColorWheelOp();
  ColorWheelOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
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