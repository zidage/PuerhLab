#include "ui/puerhlab_main/editor_dialog/modules/geometry.hpp"

#include <algorithm>

namespace puerhlab::ui::geometry {

auto ClampCropRect(float x, float y, float w, float h) -> std::array<float, 4> {
  w = std::clamp(w, kCropRectMinSize, 1.0f);
  h = std::clamp(h, kCropRectMinSize, 1.0f);
  x = std::clamp(x, 0.0f, 1.0f - w);
  y = std::clamp(y, 0.0f, 1.0f - h);
  return {x, y, w, h};
}

}  // namespace puerhlab::ui::geometry
