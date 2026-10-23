//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/ai_mask_prompt_geometry.hpp"

#include <algorithm>
#include <array>

namespace alcedo {
namespace {

[[nodiscard]] auto InsideUnitSquare(Vector2 point) -> bool {
  return point.x >= 0.0f && point.x <= 1.0f && point.y >= 0.0f && point.y <= 1.0f;
}

}  // namespace

auto MapItemToAiMaskPromptPoint(const MaskEditViewMapping& mapping, const QPointF& item)
    -> std::optional<Vector2> {
  const auto sample = MaskEditGeometry::MapItemToReference(mapping, item, /*allow_outside=*/false);
  if (!sample.has_value() || !InsideUnitSquare(sample->normalized)) {
    return std::nullopt;
  }
  return sample->normalized;
}

auto MapItemRectToAiMaskPromptBox(const MaskEditViewMapping& mapping, const QPointF& corner_a,
                                  const QPointF& corner_b) -> std::optional<NormalizedRect> {
  const std::array<QPointF, 4> corners = {corner_a, QPointF(corner_b.x(), corner_a.y()), corner_b,
                                          QPointF(corner_a.x(), corner_b.y())};
  float                        min_x   = 0.0f;
  float                        min_y   = 0.0f;
  float                        max_x   = 0.0f;
  float                        max_y   = 0.0f;
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const auto sample =
        MaskEditGeometry::MapItemToReference(mapping, corners[index], /*allow_outside=*/true);
    if (!sample.has_value()) {
      return std::nullopt;
    }
    const auto point = sample->normalized;
    if (index == 0) {
      min_x = max_x = point.x;
      min_y = max_y = point.y;
      continue;
    }
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  min_x = std::clamp(min_x, 0.0f, 1.0f);
  min_y = std::clamp(min_y, 0.0f, 1.0f);
  max_x = std::clamp(max_x, 0.0f, 1.0f);
  max_y = std::clamp(max_y, 0.0f, 1.0f);
  if (max_x <= min_x || max_y <= min_y) {
    return std::nullopt;
  }
  return NormalizedRect{min_x, min_y, max_x - min_x, max_y - min_y};
}

auto AiMaskAnalysisToItemTransform(const MaskEditViewMapping& mapping)
    -> std::optional<QTransform> {
  if (!MaskEditGeometry::IsValid(mapping)) {
    return std::nullopt;
  }
  const auto& full   = mapping.geometry.full_reference_extent;
  const auto  origin = MaskEditGeometry::MapReferenceToItem(mapping, Vector2{0.0f, 0.0f});
  const auto  x_axis =
      MaskEditGeometry::MapReferenceToItem(mapping, Vector2{static_cast<float>(full.width), 0.0f});
  const auto y_axis =
      MaskEditGeometry::MapReferenceToItem(mapping, Vector2{0.0f, static_cast<float>(full.height)});
  if (!origin.has_value() || !x_axis.has_value() || !y_axis.has_value()) {
    return std::nullopt;
  }
  // QTransform maps (u, v) to (m11 u + m21 v + dx, m12 u + m22 v + dy).
  return QTransform(x_axis->x() - origin->x(), x_axis->y() - origin->y(), y_axis->x() - origin->x(),
                    y_axis->y() - origin->y(), origin->x(), origin->y());
}

}  // namespace alcedo
