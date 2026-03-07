//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct NormalizedCropRect {
  float x_ = 0.0f;
  float y_ = 0.0f;
  float w_ = 1.0f;
  float h_ = 1.0f;
};

class CropRotateOp : public OperatorBase<CropRotateOp> {
 private:
  bool               enabled_       = false;
  float              angle_degrees_ = 0.0f;
  bool               enable_crop_   = false;
  NormalizedCropRect crop_rect_     = {};
  bool               expand_to_fit_ = true;

 public:
  static constexpr PriorityLevel     priority_level_    = 2;
  static constexpr PipelineStageName affiliation_stage_ = PipelineStageName::Geometry_Adjustment;
  static constexpr std::string_view  canonical_name_    = "Crop Rotate";
  static constexpr std::string_view  script_name_       = "crop_rotate";
  static constexpr OperatorType      operator_type_     = OperatorType::CROP_ROTATE;

  CropRotateOp() = default;
  explicit CropRotateOp(const nlohmann::json& params);

  void Apply(std::shared_ptr<ImageBuffer> input) override;
  void ApplyGPU(std::shared_ptr<ImageBuffer> input) override;

  auto GetParams() const -> nlohmann::json override;
  void SetParams(const nlohmann::json& params) override;

  void SetGlobalParams(OperatorParams& params) const override;
  void EnableGlobalParams(OperatorParams& params, bool enable) override;
};
};  // namespace puerhlab
