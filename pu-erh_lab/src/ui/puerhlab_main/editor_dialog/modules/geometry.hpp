//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>

namespace puerhlab::ui::geometry {

constexpr float kRotationSliderScale = 100.0f;
constexpr float kCropRectSliderScale = 1000.0f;
constexpr float kCropRectMinSize     = 1e-4f;

auto ClampCropRect(float x, float y, float w, float h) -> std::array<float, 4>;

}  // namespace puerhlab::ui::geometry
