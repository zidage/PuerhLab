//  Copyright 2026 Yurun Zi
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

// [WIP] New operator ODT implementation closer to ACES workflow

#pragma once

#include <edit/operators/op_base.hpp>
#include <edit/operators/utils/color_utils.hpp>

#include "type/type.hpp"

namespace puerhlab {
class ACES_ODT_Op : public OperatorBase<ACES_ODT_Op> {
 private:
  ColorUtils::ColorSpace encoding_space_ = ColorUtils::ColorSpace::REC709;  // Default to Rec.709
  ColorUtils::ETOF       encoding_etof_  = ColorUtils::ETOF::GAMMA_2_2;     // Default to Gamma 2.2

  ColorUtils::ColorSpace limiting_space_ = ColorUtils::ColorSpace::REC709;  // Default to Rec.709

  float                  peak_luminance_ = 100.0f;  // Default to 100 nits

  ColorUtils::TO_OUTPUT_Params  to_output_params_;

  static ColorUtils::ColorSpace ParseColorSpace(const std::string& cs_str);
  static ColorUtils::ETOF       ParseETOF(const std::string& etof_str);
  static std::string            ColorSpaceToString(ColorUtils::ColorSpace cs);
  static std::string            ETOFToString(ColorUtils::ETOF etof);

  void                          init_JMhParams();
  void                          init_TSParams();
  void                          init_ODTParams();

 public:
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Output_Transform;
  static constexpr OperatorType      operator_type_     = OperatorType::ODT;
  static constexpr std::string_view  canonical_name_    = "Output Device Transform (ACES)";
  static constexpr std::string_view  script_name_       = "aces_odt";

  ACES_ODT_Op()                                         = default;
  ACES_ODT_Op(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;

  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& j) override;
  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
};  // namespace puerhlab