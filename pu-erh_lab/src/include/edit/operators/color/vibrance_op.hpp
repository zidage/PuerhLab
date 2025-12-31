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

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class VibranceOp : public OperatorBase<VibranceOp> {
 private:
  /**
   * @brief An relative number for adjusting the vibrance (natural saturation)
   *
   */
  float _vibrance_offset;

  auto  ComputeScale(float chroma) -> float;

 public:
  static constexpr PriorityLevel     _priority_level    = 7;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Vibrance";
  static constexpr std::string_view  _script_name       = "vibrance";
  static constexpr OperatorType      _operator_type     = OperatorType::VIBRANCE;

  VibranceOp();
  VibranceOp(float vibrance_offset);
  VibranceOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab