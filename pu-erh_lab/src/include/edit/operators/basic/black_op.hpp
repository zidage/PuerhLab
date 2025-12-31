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

#include <json.hpp>
#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "utils/simd/simple_simd.hpp"

namespace puerhlab {
class BlackOp : public OperatorBase<BlackOp> {
 private:
  float              _offset;

  float              _y_intercept;
  float              _slope;  // slope of the tone curve

  simple_simd::f32x4 _y_intercept_vec;
  simple_simd::f32x4 _slope_vec;

 public:
  auto                               GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "BLACK";
  static constexpr std::string_view  _script_name       = "black";
  static constexpr OperatorType      _operator_type     = OperatorType::BLACK;

  BlackOp()                                             = default;
  BlackOp(float offset);
  BlackOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;

  void        SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab