//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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

  void ReleaseThumbnail(sl_element_id_t sleeve_element_id);
};
};  // namespace puerhlab