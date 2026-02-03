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

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

class TintOp : public OperatorBase<TintOp> {
 private:
  /**
   * @brief An relative number for adjusting the tint,
   * negative toward green, positive toward magneta
   * Range from -100 to 100
   */
  float tint_offset_;

  /**
   * @brief An absolute number for adjusting the tint,
   * negative toward green, positive toward magneta
   * Range from -1.0f to 1.0f
   */
  float scale_;

 public:
  static constexpr PriorityLevel     priority_level_    = 4;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Color_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Tint";
  static constexpr std::string_view  script_name_       = "tint";
  static constexpr OperatorType      operator_type_     = OperatorType::TINT;
  TintOp();
  TintOp(float tint_offset);
  TintOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
}  // namespace puerhlab