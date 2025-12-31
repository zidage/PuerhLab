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
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {
  float normalized_offset = _offset / 100.0f;
  _gamma                  = std::pow(2.0f, -normalized_offset * 1.3f);
}

ShadowsOp::ShadowsOp(const nlohmann::json& params) {
  SetParams(params);
  float normalized_offset = _offset / 100.0f;
  _gamma                  = std::pow(2.0f, -normalized_offset * 1.3f);
}

auto ShadowsOp::GetScale() -> float { return _offset / 100.0f; }

void ShadowsOp::Apply(std::shared_ptr<ImageBuffer> input) {
  {
  }
}

static inline float Luma(const Pixel& rgb) {
  return 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
}


auto ShadowsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void ShadowsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset        = params[_script_name].get<float>();
    _curve.control = std::clamp(_offset / 100.0f, -1.0f, 1.0f);
    _curve.toe_end = std::clamp(0.55f, 0.0f, 1.0f);
    _curve.m0      = 1.0f + _curve.control * _curve.slope_range;
    _curve.x1      = _curve.toe_end;
    _curve.y1      = _curve.x1;
    _curve.dx      = _curve.x1 - _curve.x0;
  }
}

void ShadowsOp::SetGlobalParams(OperatorParams& params) const {
  params.shadows_offset = _offset / 100.0f;
  params.shadows_m0     = 1.0f + params.shadows_offset * _curve.slope_range;
  params.shadows_x0     = _curve.x0;
  params.shadows_x1     = _curve.x1;
  params.shadows_y0     = _curve.y0;
  params.shadows_y1     = _curve.y1;
  params.shadows_m1     = _curve.m1;
  params.shadows_dx     = _curve.dx;
}
}  // namespace puerhlab