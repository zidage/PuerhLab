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

class ContrastOp : public OperatorBase<ContrastOp> {
 private:
  /**
   * @brief A relative number for adjusting the image
   *
   */
  float contrast_offset_;
  /**
   * @brief An absolute number to represent the contrast after adjustment
   * Usually, it is computed through dividing 100.0f from the offset
   *
   */
  float scale_;

 public:
  static constexpr PriorityLevel     priority_level_    = 3;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Contrast";
  static constexpr std::string_view  script_name_       = "contrast";
  static constexpr OperatorType      operator_type_     = OperatorType::CONTRAST;
  ContrastOp();
  ContrastOp(float contrast_offset);
  ContrastOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;
  void SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab