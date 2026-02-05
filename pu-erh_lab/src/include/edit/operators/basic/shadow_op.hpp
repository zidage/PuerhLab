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
  float       control_;             // [-1, 1]
  float       toe_end_     = 0.25;  // end of toe region in [0,1], e.g. 0.25
  const float slope_range_ = 0.8f;

  // Hermite between x0=0 and x1=toe_end
  float       m0_;         // slope at blackpoint (x0)
  float       m1_ = 1.0f;  // slope at x1 to keep continuity

  float       x0_ = 0.0f;
  float       x1_;         // = toe_end
  float       y0_ = 0.0f;  // identity at x0
  float       y1_;         // = x1, identity at x1

  float       dx_;  // x1 - x0
};
class ShadowsOp : public OperatorBase<ShadowsOp> {
 private:
  float                offset_;
  ShadowCurveParams    curve_{};

  static constexpr int kLutSize = 1024;
  std::vector<float>   lut_;

  float                gamma_;
  float                inv_threshold_ = 1.0f / 0.5f;

  void                 InitializeLUT();

 public:
  auto                               GetScale() -> float;
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Shadows";
  static constexpr std::string_view  script_name_       = "shadows";
  static constexpr OperatorType      operator_type_     = OperatorType::SHADOWS;
  ShadowsOp()                                           = default;
  ShadowsOp(float offset);
  ShadowsOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  void        ApplyGPU(std::shared_ptr<ImageBuffer> input) override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;

  void        SetGlobalParams(OperatorParams& params) const override;
  void        EnableGlobalParams(OperatorParams& params, bool enable) override;
};
}  // namespace puerhlab