//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include "sidecar_client/dto/segmentation.hpp"
#include "sidecar_client/segmentation_client.hpp"

namespace alcedo {

/// Timeouts of the segmentation calls. A cold DirectML encoder load takes about 3 s; the Core ML
/// models can need longer on the first load after an operating system update.
struct AiMaskSegmentationTimeouts {
  std::chrono::milliseconds load{120000};
  std::chrono::milliseconds start{30000};
  std::chrono::milliseconds decode{10000};
  std::chrono::milliseconds control{5000};
};

/**
 * @brief One activation of the AI Mask tool over the sidecar segmentation service (SM3.A).
 *
 * The service holds a sidecar lease, so the sidecar process keeps running while the tool is
 * open, and one segmentation client with its own persistent channel. It owns the identity of the
 * current segmentation session; the sidecar owns the models and the image embedding. The caller
 * owns the selection and passes the full prompt to Decode().
 *
 * Close() ends the session, unloads the models (lab decision U6: the models unload when the user
 * goes back to adjusting), and then releases the lease. The destructor calls Close().
 *
 * Threading: not thread-safe. The owner calls every method from one thread. Each method blocks
 * until the sidecar answers or its timeout passes.
 */
class AiMaskSegmentationService {
 public:
  /// @p sidecar_lease keeps the sidecar process running (AiSidecarRuntimeService::AcquireLease).
  /// @p client talks to that process.
  AiMaskSegmentationService(std::shared_ptr<void>                               sidecar_lease,
                            std::unique_ptr<sidecar_client::SegmentationClient> client,
                            AiMaskSegmentationTimeouts                          timeouts = {});
  ~AiMaskSegmentationService();

  AiMaskSegmentationService(const AiMaskSegmentationService&)                    = delete;
  auto operator=(const AiMaskSegmentationService&) -> AiMaskSegmentationService& = delete;

  /// Loads the model set from @p model_directory on @p device ("" for the platform default).
  auto LoadModels(const std::filesystem::path& model_directory, const std::string& device = {})
      -> SegmentationCallStatus;
  /// Ends the current session, then starts a session on @p image. When the end fails, no session
  /// starts and the end failure is returned.
  auto               StartImage(const SegmentationRgb8Image& image) -> SegmentationSessionResult;
  /// Decodes @p prompt on the current session. Fails without a sidecar call when no session
  /// exists.
  auto               Decode(const SegmentationPrompt& prompt) -> SegmentationDecodeResult;
  /// Ends the current session. Succeeds without a sidecar call when no session exists.
  auto               EndImage() -> SegmentationCallStatus;
  /// Ends the session, unloads the models, and releases the lease. Later calls fail without a
  /// sidecar call. A second Close() succeeds without a call. Returns the first failure.
  auto               Close() -> SegmentationCallStatus;

  [[nodiscard]] auto HasSession() const -> bool { return !session_id_.empty(); }
  [[nodiscard]] auto IsClosed() const -> bool { return closed_; }

 private:
  [[nodiscard]] static auto LocalFailure(std::string message) -> SegmentationCallStatus;

  std::shared_ptr<void>     sidecar_lease_;
  std::unique_ptr<sidecar_client::SegmentationClient> client_;
  AiMaskSegmentationTimeouts                          timeouts_;
  std::string                                         session_id_;
  bool                                                closed_ = false;
};

}  // namespace alcedo
