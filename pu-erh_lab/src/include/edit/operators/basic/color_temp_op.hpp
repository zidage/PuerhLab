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

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class ColorTempOp : public OperatorBase<ColorTempOp> {
 private:
  ColorTempMode mode_         = ColorTempMode::AS_SHOT;
  float         custom_cct_   = 6500.0f;
  float         custom_tint_  = 0.0f;
  mutable float resolved_cct_ = 6500.0f;
  mutable float resolved_tint_ = 0.0f;

  static auto   ParseMode(const std::string& mode) -> ColorTempMode;
  static auto   ModeToString(ColorTempMode mode) -> std::string;

 public:
  static constexpr PriorityLevel     priority_level_    = 0;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::To_WorkingSpace;
  static constexpr OperatorType      operator_type_     = OperatorType::COLOR_TEMP;
  static constexpr std::string_view  canonical_name_    = "Color Temperature";
  static constexpr std::string_view  script_name_       = "color_temp";

  ColorTempOp()                                          = default;
  explicit ColorTempOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;

  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;

  // Resolve runtime camera -> XYZ/AP1 matrices from current RAW context.
  void ResolveRuntime(OperatorParams& params) const;
};
}  // namespace puerhlab
