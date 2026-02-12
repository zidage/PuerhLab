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

#include <array>
#include <opencv2/core.hpp>

#include "edit/operators/op_base.hpp"
#include "type/type.hpp"

namespace puerhlab {

class HLSOp : public OperatorBase<HLSOp> {
 private:
  static constexpr int kHueProfileCount = OperatorParams::kHlsProfileCount;

  cv::Vec3f target_hls_;

  cv::Vec3f hls_adjustment_;

  float     hue_range_;
  float     lightness_range_;
  float     saturation_range_;

  std::array<float, kHueProfileCount>    hue_profile_values_{};
  std::array<cv::Vec3f, kHueProfileCount> hls_adjustment_table_{};
  std::array<float, kHueProfileCount>    hue_range_table_{};
  int                                    active_profile_idx_ = 0;

 public:
  static constexpr PriorityLevel     priority_level_    = 5;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  canonical_name_    = "HLS";
  static constexpr std::string_view  script_name_       = "HLS";
  static constexpr OperatorType      operator_type_     = OperatorType::HLS;
  HLSOp();
  HLSOp(const nlohmann::json& params);

  void SetTargetColor(const cv::Vec3f& bgr_color_normalized);
  void SetAdjustment(const cv::Vec3f& adjustment);
  void SetRanges(float h_range, float l_range, float s_range);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
};  // namespace puerhlab
