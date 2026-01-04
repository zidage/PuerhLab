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
#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
class WhiteOp : public OperatorBase<WhiteOp> {
 private:
  float              offset_;

  float              y_intercept_;
  float              black_point_;
  float              slope_;


 public:
  auto                               GetScale() -> float;
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  canonical_name_    = "WHITE";
  static constexpr std::string_view  script_name_       = "white";
  static constexpr OperatorType      operator_type_     = OperatorType::WHITE;
  WhiteOp()                                             = default;
  WhiteOp(float offset);
  WhiteOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;

  void        SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab