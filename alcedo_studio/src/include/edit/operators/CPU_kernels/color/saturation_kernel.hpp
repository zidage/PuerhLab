//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "edit/operators/op_base.hpp"

namespace alcedo {
struct SaturationOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.saturation_enabled_) return;

    float luma = 0.2126f * p.r_ + 0.7152f * p.g_ + 0.0722f * p.b_;
    p.r_        = luma + (p.r_ - luma) * params.saturation_offset_;
    p.g_        = luma + (p.g_ - luma) * params.saturation_offset_;
    p.b_        = luma + (p.b_ - luma) * params.saturation_offset_;
  }
};
}  // namespace alcedo