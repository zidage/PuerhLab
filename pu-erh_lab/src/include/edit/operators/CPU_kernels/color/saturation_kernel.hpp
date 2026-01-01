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
struct SaturationOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.saturation_enabled_) return;

    float luma = 0.2126f * p.r_ + 0.7152f * p.g_ + 0.0722f * p.b_;
    p.r_        = luma + (p.r_ - luma) * params.saturation_offset_;
    p.g_        = luma + (p.g_ - luma) * params.saturation_offset_;
    p.b_        = luma + (p.b_ - luma) * params.saturation_offset_;
  }
};
}  // namespace puerhlab