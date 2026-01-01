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
struct ROI {
  int   x_;
  int   y_;
  float resize_factor_;
};

class ResizeOp : public OperatorBase<ResizeOp> {
 private:
  int  maximum_edge_;

  bool enable_roi_ = false;
  ROI  roi_;

 public:
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Image_Loading;
  static constexpr std::string_view  canonical_name_    = "Resize";
  static constexpr std::string_view  script_name_       = "resize";
  static constexpr OperatorType      operator_type_     = OperatorType::RESIZE;
  ResizeOp()                                            = default;
  ResizeOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
};
};  // namespace puerhlab