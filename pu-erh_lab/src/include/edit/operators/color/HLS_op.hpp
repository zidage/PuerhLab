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

#include <opencv2/core.hpp>

#include "edit/operators/op_base.hpp"
#include "type/type.hpp"

namespace puerhlab {

class HLSOp : public OperatorBase<HLSOp> {
 private:
  cv::Vec3f _target_HLS;

  cv::Vec3f _HLS_adjustment;

  float     _hue_range;
  float     _lightness_range;
  float     _saturation_range;

 public:
  static constexpr PriorityLevel     _priority_level    = 5;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  _canonical_name    = "HLS";
  static constexpr std::string_view  _script_name       = "HLS";
  static constexpr OperatorType      _operator_type     = OperatorType::HLS;

  HLSOp();
  HLSOp(const nlohmann::json& params);

  void SetTargetColor(const cv::Vec3f& bgr_color_normalized);
  void SetAdjustment(const cv::Vec3f& adjustment);
  void SetRanges(float h_range, float l_range, float s_range);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
};
};  // namespace puerhlab