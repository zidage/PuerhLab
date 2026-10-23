//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QPointF>
#include <QTransform>
#include <optional>

#include "edit/geometry/types.hpp"
#include "ui/edit_viewer/mask_edit_geometry.hpp"

/**
 * @file
 * @brief Viewport input to prompt coordinates of the AI Mask analysis image, and back (SM4.2).
 *
 * The user places prompts on the photo as the viewport presents it: after the document crop and
 * rotation, or in the Geometry panel frame, at any zoom and pan, and on a region-of-interest
 * patch. The analysis image is the whole source without crop and rotation, so its normalized
 * coordinates are normalized ReferenceSpace coordinates. The functions compose
 * @ref MaskEditGeometry with that rule. GUI-thread geometry only.
 */
namespace alcedo {

/**
 * @brief Normalized analysis image coordinates of the pointer at @p item (SM4 plan S6).
 *
 * @return Empty when @p mapping is invalid, @p item is outside the displayed photograph, or the
 *         position is outside the source (for example a border corner of the Geometry panel
 *         frame). A position on the source edge is inside.
 */
[[nodiscard]] auto MapItemToAiMaskPromptPoint(const MaskEditViewMapping& mapping,
                                              const QPointF& item) -> std::optional<Vector2>;

/**
 * @brief Normalized analysis image box of the item rectangle with the opposite corners
 *        @p corner_a and @p corner_b (SM4 plan S7).
 *
 * The four item corners map to ReferenceSpace. With a document rotation they form a rotated
 * rectangle; the box is their axis-aligned bounding box, clipped to the source. The corners can
 * lie outside the displayed photograph.
 *
 * @return Empty when @p mapping is invalid or the clipped box has no area.
 */
[[nodiscard]] auto MapItemRectToAiMaskPromptBox(const MaskEditViewMapping& mapping,
                                                const QPointF& corner_a, const QPointF& corner_b)
    -> std::optional<NormalizedRect>;

/**
 * @brief Affine transform from normalized analysis image coordinates to item coordinates
 *        (SM4 plan S8).
 *
 * Draws a Candidate Mask of the analysis image over the presented photo. The transform is valid
 * for @p mapping only; a change of the view mapping needs a new transform.
 *
 * @return Empty when @p mapping is invalid.
 */
[[nodiscard]] auto AiMaskAnalysisToItemTransform(const MaskEditViewMapping& mapping)
    -> std::optional<QTransform>;

}  // namespace alcedo
