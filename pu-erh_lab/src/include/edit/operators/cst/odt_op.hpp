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

#pragma once

#include "edit/operators/cst/open_drt_cpu.hpp"
#include "edit/operators/op_base.hpp"

namespace puerhlab {

class ODT_Op : public OperatorBase<ODT_Op> {
 private:
  ColorUtils::ODTMethod      method_           = ColorUtils::ODTMethod::OPEN_DRT;
  ColorUtils::ColorSpace     encoding_space_   = ColorUtils::ColorSpace::REC709;
  ColorUtils::ETOF           encoding_etof_    = ColorUtils::ETOF::GAMMA_2_2;
  ColorUtils::ColorSpace     limiting_space_   = ColorUtils::ColorSpace::REC709;
  float                      peak_luminance_   = 100.0f;
  odt_cpu::OpenDRTSettings   open_drt_settings_ = {};
  ColorUtils::TO_OUTPUT_Params to_output_params_ = {};

  void RebuildRuntime();
  void ValidateParams() const;

 public:
  static constexpr PriorityLevel    priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Output_Transform;
  static constexpr OperatorType     operator_type_     = OperatorType::ODT;
  static constexpr std::string_view canonical_name_    = "Output Device Transform";
  static constexpr std::string_view script_name_       = "odt";

  ODT_Op() = default;
  explicit ODT_Op(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;

  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& j) override;
  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};

}  // namespace puerhlab
