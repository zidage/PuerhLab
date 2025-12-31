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

struct ShadowCurveParams {
  float       control;             // [-1, 1]
  float       toe_end     = 0.25;  // end of toe region in [0,1], e.g. 0.25
  const float slope_range = 0.8f;

  // Hermite between x0=0 and x1=toe_end
  float       m0;         // slope at blackpoint (x0)
  float       m1 = 1.0f;  // slope at x1 to keep continuity

  float       x0 = 0.0f;
  float       x1;         // = toe_end
  float       y0 = 0.0f;  // identity at x0
  float       y1;         // = x1, identity at x1

  float       dx;  // x1 - x0
};
class ShadowsOp : public OperatorBase<ShadowsOp> {
 private:
  float                _offset;
  ShadowCurveParams    _curve{};

  static constexpr int kLutSize = 1024;
  std::vector<float>   _lut;

  float                _gamma;
  float                _inv_threshold = 1.0f / 0.5f;

  void                 InitializeLUT();

 public:
  auto                               GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Shadows";
  static constexpr std::string_view  _script_name       = "shadows";
  static constexpr OperatorType      _operator_type     = OperatorType::SHADOWS;

  ShadowsOp()                                           = default;
  ShadowsOp(float offset);
  ShadowsOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;

  void        SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab