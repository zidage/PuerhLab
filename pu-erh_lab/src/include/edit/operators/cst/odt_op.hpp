//  Copyright 2026 Yurun Zi
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

#include <edit/operators/op_base.hpp>
#include <edit/operators/utils/color_utils.hpp>

#include "type/type.hpp"

namespace puerhlab {
class OutputTransformOp : public OperatorBase<OutputTransformOp> {
 private:
  nlohmann::json                authoring_params_ = nlohmann::json::object();
  ColorUtils::TO_OUTPUT_Params  to_output_params_ = {};

  static ColorUtils::ColorSpace ParseColorSpace(const std::string& cs_str);
  static ColorUtils::ETOF       ParseETOF(const std::string& etof_str);
  static ColorUtils::OutputTransformMethod ParseMethod(const std::string& method_str);
  static std::string            ColorSpaceToString(ColorUtils::ColorSpace cs);
  static std::string            ETOFToString(ColorUtils::ETOF etof);
  static std::string            MethodToString(ColorUtils::OutputTransformMethod method);

  void                          ResolveOutputTransform();
  void                          ResolveACESParams();
  void                          ResolveOpenDRTParams();

 public:
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Output_Transform;
  static constexpr OperatorType      operator_type_     = OperatorType::ODT;
  static constexpr std::string_view  canonical_name_    = "Output Device Transform";
  static constexpr std::string_view  script_name_       = "odt";

  OutputTransformOp();
  explicit OutputTransformOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;

  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& j) override;
  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
}  // namespace puerhlab
