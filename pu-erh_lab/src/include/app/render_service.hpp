//  Copyright 2025 Yurun Zi
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

#include <memory>
#include <vector>

#include "renderer/pipeline_scheduler.hpp"

namespace puerhlab {
class RenderService {
 public:
  static auto GetPreviewScheduler() -> std::shared_ptr<PipelineScheduler> {
    return std::make_shared<PipelineScheduler>();
  }

  static auto GetThumbnailOrExportScheduler() -> std::shared_ptr<PipelineScheduler> {
    static size_t thread_count = fmax(size_t(2), std::thread::hardware_concurrency() / 2);
    static std::shared_ptr<PipelineScheduler> scheduler =
        std::make_shared<PipelineScheduler>(thread_count);
    return scheduler;
  }

  static auto GetThumbnailOrExportScheduler(size_t thread_cnt)
      -> std::shared_ptr<PipelineScheduler> {
    return std::make_shared<PipelineScheduler>(thread_cnt);
  }
};
};  // namespace puerhlab