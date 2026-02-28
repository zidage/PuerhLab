#pragma once

namespace puerhlab::ui::color_temp {

constexpr int   kCctMin        = 2000;
constexpr int   kCctMax        = 15000;
constexpr int   kTintMin       = -150;
constexpr int   kTintMax       = 150;
constexpr int   kSliderUiMin   = 0;
constexpr int   kSliderUiMax   = 4096;
constexpr int   kSliderUiMid   = 2048;
constexpr float kPivotCct      = 6000.0f;

auto SliderPosToCct(int pos) -> float;
auto CctToSliderPos(float cct) -> int;

}  // namespace puerhlab::ui::color_temp
