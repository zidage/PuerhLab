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
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct InputMeta {
  std::string cam_maker_;
  std::string cam_model_;

  std::string lens_maker_;
  std::string lens_model_;

  float       focal_length_mm_;
  float       aperture_f_number_;
  float       distance_m_;
  float       sensor_width_mm_;  // optional; used only for fallback crop-factor estimation
};

class LensCalibOp : public OperatorBase<LensCalibOp> {
 private:
  InputMeta             input_meta_;

  std::filesystem::path lens_profile_db_path_;

  bool                  enabled_ = true;

 public:
  static constexpr PriorityLevel     priority_level_    = 1;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Geometry_Adjustment;
  static constexpr std::string_view  canonical_name_    = "LensCalibration";
  static constexpr std::string_view  script_name_       = "lens_calib";
  static constexpr OperatorType      operator_type_     = OperatorType::LENS_CALIBRATION;

  LensCalibOp()                                         = default;
  LensCalibOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;
  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
}  // namespace puerhlab