//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <functional>
#include <memory>

#include "app/image_pool_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/sleeve_service.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct ThumbnailGuard {
  std::unique_ptr<ImageBuffer> thumbnail_buffer_ = nullptr;
  int                          pin_count_        = 0;

  ThumbnailGuard()  = default;
  ~ThumbnailGuard() = default;

  // Non-copyable
  ThumbnailGuard(const ThumbnailGuard&)            = delete;
  ThumbnailGuard& operator=(const ThumbnailGuard&) = delete;

  // Movable
  ThumbnailGuard(ThumbnailGuard&&)            = default;
  ThumbnailGuard& operator=(ThumbnailGuard&&) = default;
};

using ThumbnailCallback  = std::function<void(std::shared_ptr<ThumbnailGuard>)>;
using CallbackDispatcher = std::function<void(std::function<void()>)>;

class ThumbnailService {
 private:
  struct State;
  std::shared_ptr<State> state_;

  static void            HandleEvict(State& st, std::optional<sl_element_id_t> evicted_id);

 public:
  ThumbnailService() = delete;
  ThumbnailService(std::shared_ptr<SleeveServiceImpl>   sleeve_service,
                   std::shared_ptr<ImagePoolService>    image_pool_service,
                   std::shared_ptr<PipelineMgmtService> pipeline_service);
  ~ThumbnailService() = default;

  void GetThumbnail(sl_element_id_t id, image_id_t image_id, ThumbnailCallback callback, bool pin_if_found = true,
                    CallbackDispatcher dispatcher = nullptr);

  // Force the cached thumbnail for this sleeve element to be discarded.
  // Next GetThumbnail() will re-render via pipeline.
  void InvalidateThumbnail(sl_element_id_t sleeve_element_id);

  void ReleaseThumbnail(sl_element_id_t sleeve_element_id);
};
};  // namespace puerhlab