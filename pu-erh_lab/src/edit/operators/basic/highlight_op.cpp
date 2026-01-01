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

#include "edit/operators/basic/highlight_op.hpp"

#include <cmath>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
HighlightsOp::HighlightsOp(float offset) : offset_(offset) {}

HighlightsOp::HighlightsOp(const nlohmann::json& params) { SetParams(params); }

static inline float clampf(float v, float a, float b) { return std::max(a, std::min(b, v)); }

// Luminance (linear, BGR)

auto                HighlightsOp::GetScale() -> float { return offset_ / 300.0f; }

void                HighlightsOp::Apply(std::shared_ptr<ImageBuffer> input) { (void)input; }


auto HighlightsOp::GetParams() const -> nlohmann::json { return {script_name_, offset_}; }

void HighlightsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) {
    offset_ = 0.0f;
  } else {
    offset_ = params[script_name_].get<float>();
  }
  curve_.control_    = clampf(offset_ / 100.0f, -1.0f, 1.0f);
  curve_.knee_start_ = clampf(0.2f, 0.0f, 1.0f);  // ensure <= whitepoint
  // map control -> slope at whitepoint (m1)
  // design: control = +1 => strong compression (m1 -> small, e.g. 0.2)
  //         control =  0 => identity slope (1.0)
  //         control = -1 => boost highlights (m1 -> >1, e.g. 1.8)
  curve_.m1_ = 1.0f - curve_.control_ * curve_.slope_range_;  // in [1-slope_range, 1+slope_range]

  // endpoints for Hermite between x0 = knee_start, x1 = whitepoint
  curve_.x0_ = curve_.knee_start_;
  curve_.y0_ = curve_.x0_;  // keep continuity (identity at x0)
  curve_.y1_ = curve_.x1_;  // identity at x1 (we'll control slope to shape shoulder)

  // For Hermite formula we need derivatives dy/dx at endpoints.
  // m0 and m1 are dy/dx at x0 and x1 respectively.
  // But Hermite cubic uses tangents scaled by (x1-x0) in the basis:
  curve_.dx_ = (curve_.x1_ - curve_.x0_);
}

void HighlightsOp::SetGlobalParams(OperatorParams& params) const {
  params.highlights_offset_ = offset_ / 100.0f;
  params.highlights_m1_     = curve_.m1_;
}
}  // namespace puerhlab