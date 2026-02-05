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

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

/**
 * @brief Already deprecated. Use CSTOp instead.
 *
 */
class CVCvtColorOp : public OperatorBase<CVCvtColorOp> {
 private:
  int                   code_;
  std::optional<size_t> channel_index_;

 public:
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  canonical_name_    = "CV CvtColor";
  static constexpr std::string_view  script_name_       = "CV_CvtColor";

  CVCvtColorOp()                                        = default;
  CVCvtColorOp(int code, std::optional<size_t> channel_index);
  CVCvtColorOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
}  // namespace puerhlab