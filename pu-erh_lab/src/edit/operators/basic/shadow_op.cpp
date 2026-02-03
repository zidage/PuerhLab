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

#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_kernel.hpp"
#include "edit/operators/utils/functions.hpp"
#include "hwy/contrib/math/math-inl.h"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : offset_(offset) {
  float normalized_offset = offset_ / 100.0f;
  gamma_                  = std::pow(2.0f, -normalized_offset * 1.3f);
}

ShadowsOp::ShadowsOp(const nlohmann::json& params) {
  SetParams(params);
  float normalized_offset = offset_ / 100.0f;
  gamma_                  = std::pow(2.0f, -normalized_offset * 1.3f);
}

auto ShadowsOp::GetScale() -> float { return offset_ / 100.0f; }

void ShadowsOp::Apply(std::shared_ptr<ImageBuffer> input) {
  {
  }
}

static inline float Luma(const Pixel& rgb) {
  return 0.2126f * rgb.r_ + 0.7152f * rgb.g_ + 0.0722f * rgb.b_;
}


auto ShadowsOp::GetParams() const -> nlohmann::json { return {script_name_, offset_}; }

void ShadowsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) {
    offset_ = 0.0f;
  } else {
    offset_        = params[script_name_].get<float>();
    curve_.control_ = std::clamp(offset_ / 80.0f, -1.0f, 1.0f);
    curve_.toe_end_ = std::clamp(0.55f, 0.0f, 1.0f);
    curve_.m0_      = 1.0f + curve_.control_ * curve_.slope_range_;
    curve_.x1_      = curve_.toe_end_;
    curve_.y1_      = curve_.x1_;
    curve_.dx_      = curve_.x1_ - curve_.x0_;
  }
}

void ShadowsOp::SetGlobalParams(OperatorParams& params) const {
  params.shadows_offset_ = offset_ / 80.0f;
  params.shadows_m0_     = 1.0f + params.shadows_offset_ * curve_.slope_range_;
  params.shadows_x0_     = curve_.x0_;
  params.shadows_x1_     = curve_.x1_;
  params.shadows_y0_     = curve_.y0_;
  params.shadows_y1_     = curve_.y1_;
  params.shadows_m1_     = curve_.m1_;
  params.shadows_dx_     = curve_.dx_;
}

void ShadowsOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.shadows_enabled_ = enable;
}
}  // namespace puerhlab