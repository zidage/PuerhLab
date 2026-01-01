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
#include "image/image_buffer.hpp"

namespace puerhlab {
class ClarityOp : public OperatorBase<ClarityOp> {
 private:
  /**
   * @brief Offset to the clarity of the image, ranging from -100 to 100
   *
   */
  float        clarity_offset_;
  /**
   * @brief Scaled offset to the clarity, ranging from -1.0f to 1.0f
   *
   */
  float        scale_;

  /**
   * @brief An internal-use-only parameter to adjust the radius of the USM sharpening filter
   *
   */
  static float usm_radius_;

  void         CreateMidtoneMask(cv::Mat& input, cv::Mat& mask) const;

 public:
  static constexpr PriorityLevel     priority_level_    = 8;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Detail_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Clarity";
  static constexpr std::string_view  script_name_       = "clarity";
  static constexpr OperatorType      operator_type_     = OperatorType::CLARITY;
  ClarityOp();
  ClarityOp(float clarity_offset);
  ClarityOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab