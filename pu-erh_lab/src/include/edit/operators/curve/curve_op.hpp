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

#include <opencv2/core/types.hpp>
#include <vector>

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class CurveOp : public OperatorBase<CurveOp> {
 private:
  std::vector<cv::Point2f> ctrl_pts_{};
  std::vector<float>       h_{};
  std::vector<float>       m_{};

  void                     ComputeTagents();
  auto                     EvaluateCurve(float x) const -> float;

 public:
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Curve";
  static constexpr std::string_view  script_name_       = "curve";
  static constexpr OperatorType      operator_type_     = OperatorType::CURVE;
  CurveOp()                                             = delete;
  CurveOp(const std::vector<cv::Point2f>& control_points);
  CurveOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void SetCtrlPts(const std::vector<cv::Point2f>& control_points);
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
};  // namespace puerhlab
