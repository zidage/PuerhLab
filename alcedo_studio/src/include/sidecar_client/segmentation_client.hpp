//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "sidecar_client/dto/segmentation.hpp"

namespace alcedo::sidecar_client {

/**
 * @brief Client of the sidecar `SegmentationService` (AI Mask, SM3.A).
 *
 * Each call blocks until the sidecar answers or the timeout passes. A failed operation returns
 * `ok = false` with the real error of the sidecar or of the transport. The client keeps no
 * segmentation state; the sidecar owns the models and the sessions.
 */
class SegmentationClient {
 public:
  virtual ~SegmentationClient()                                                          = default;

  /// Loads the encoder and the decoder from @p model_directory on @p device ("" for the platform
  /// default).
  virtual auto LoadModels(const std::string& model_directory, const std::string& device,
                          std::chrono::milliseconds timeout) -> SegmentationCallStatus   = 0;
  /// Ends all sessions and unloads the models.
  virtual auto UnloadModels(std::chrono::milliseconds timeout) -> SegmentationCallStatus = 0;
  virtual auto GetStatus(std::chrono::milliseconds timeout) -> SegmentationStatus        = 0;
  /// Runs the encoder once on @p image and returns the new session identity.
  virtual auto StartSession(const SegmentationRgb8Image& image, std::chrono::milliseconds timeout)
      -> SegmentationSessionResult                                                   = 0;
  /// Runs the decoder once on the session embedding with the full prompt of the selection.
  virtual auto Decode(const std::string& session_id, const SegmentationPrompt& prompt,
                      std::chrono::milliseconds timeout) -> SegmentationDecodeResult = 0;
  virtual auto EndSession(const std::string& session_id, std::chrono::milliseconds timeout)
      -> SegmentationCallStatus = 0;
};

/**
 * @brief Makes a gRPC segmentation client for the sidecar at @p endpoint ("host:port").
 *
 * The client creates one channel and one stub, and uses them for every call of its lifetime, so
 * all calls share one connection. The send and receive limits are 64 MB: a 4096 x 4096 RGB8 image
 * is 50.3 MB. The channel connects on the first call.
 */
auto MakeGrpcSegmentationClient(std::string endpoint) -> std::unique_ptr<SegmentationClient>;

}  // namespace alcedo::sidecar_client
