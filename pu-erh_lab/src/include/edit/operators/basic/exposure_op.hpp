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

#include <hwy/highway.h>

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "type/type.hpp"
#include "utils/simd/simple_simd.hpp"

namespace puerhlab {
class ExposureOp : public OperatorBase<ExposureOp>, PointOpTag {
 private:
  /**
   * @brief An EV offset applied to the target image.
   * Positive to increase the brightness, negative to darken.
   *
   */
  float              _exposure_offset;

  /**
   * @brief The actual luminance offset derived from the EV
   * dL = 2^E
   *
   */
  float              _scale;

  float              _gamma;


 public:
  static constexpr PriorityLevel     _priority_level    = 0;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Exposure";
  static constexpr std::string_view  _script_name       = "exposure";
  static constexpr OperatorType      _operator_type     = OperatorType::EXPOSURE;
  ExposureOp();
  ExposureOp(float exposure_offset);
  ExposureOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
};
}  // namespace puerhlab