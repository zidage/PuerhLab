//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace alcedo {

/// Outcome of one segmentation call. `status` and `error_code` are the `alcedo.ai`
/// AiResponseStatus and AiErrorCode values; a transport failure sets `ok = false` with the gRPC
/// message and leaves `status` at AI_STATUS_UNSPECIFIED.
struct SegmentationCallStatus {
  bool        ok         = false;
  int         status     = 0;
  int         error_code = 0;
  std::string error;
};

/// An RGB8 image for a segmentation session: row-major, 3 bytes per pixel, long edge at most
/// 4096 px. A start request from the image embedding cache only sends an empty image (width 0,
/// height 0, no pixels).
struct SegmentationRgb8Image {
  uint32_t             width  = 0;
  uint32_t             height = 0;
  std::vector<uint8_t> pixels;
};

enum class SegmentationPointLabel : uint8_t {
  kSubtract = 0,
  kAdd      = 1,
};

/// A prompt point in normalized image coordinates [0, 1].
struct SegmentationPoint {
  float                  x     = 0.0F;
  float                  y     = 0.0F;
  SegmentationPointLabel label = SegmentationPointLabel::kAdd;
};

/// A prompt box in normalized image coordinates; the corners can come in any order.
struct SegmentationBox {
  float x0 = 0.0F;
  float y0 = 0.0F;
  float x1 = 0.0F;
  float y1 = 0.0F;
};

/// The decoder input of one stroke, click, or box: all points and the box of the selection, and
/// the selected logits of the previous decode (empty for the first decode).
struct SegmentationPrompt {
  std::vector<SegmentationPoint> points;
  std::optional<SegmentationBox> box;
  std::vector<float>             mask_input;
};

/// One decoder mask token: logit_size x logit_size logits clipped to +-32.
struct SegmentationMaskLevel {
  std::vector<float> logits;
  float              score = 0.0F;
  uint32_t           token = 0;
  uint32_t           area  = 0;
};

/// Result of a decode. A click returns three levels sorted by area; other prompts return one
/// mask.
struct SegmentationDecodeResult {
  SegmentationCallStatus             status;
  std::vector<SegmentationMaskLevel> masks;
  uint32_t                           selected_index = 0;
  uint32_t                           logit_size     = 0;
  int64_t                            decoder_ms     = 0;
};

struct SegmentationSessionResult {
  SegmentationCallStatus status;
  std::string            session_id;
  /// Time of the preprocessing and the encoder; 0 for a cached image.
  int64_t                encoder_ms       = 0;
  /// True when the session uses the cached image embedding of the image key.
  bool                   from_cache       = false;
  /// True when a start without pixels names an image key that the sidecar does not cache
  /// (AI_ERROR_IMAGE_NOT_CACHED). The caller renders the image and starts again with the pixels.
  bool                   image_not_cached = false;
};

struct SegmentationModelResidency {
  std::string role;
  std::string device;
  int64_t     load_ms = 0;
};

struct SegmentationStatus {
  SegmentationCallStatus                  status;
  bool                                    loaded = false;
  std::string                             model_set_id;
  std::string                             model_directory;
  std::string                             device;
  std::string                             backend;
  std::vector<SegmentationModelResidency> models;
  uint32_t                                session_count      = 0;
  uint32_t                                cached_image_count = 0;
};

}  // namespace alcedo
