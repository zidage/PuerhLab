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

#include <memory>
#include <queue>

#include "app/sleeve_service.hpp"
#include "concurrency/thread_pool.hpp"
#include "image_pool_service.hpp"
#include "pipeline_service.hpp"
#include "render_service.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "type/supported_file_type.hpp"
#include "type/type.hpp"

namespace puerhlab {


struct ExportTask {
  sl_element_id_t sleeve_id_;
  image_id_t      image_id_;
  ExportFormatOptions   options_;
};

struct ExportResult {
  bool        success_ = false;
  std::string message_;
};

class ExportService {
 private:
  std::shared_ptr<SleeveServiceImpl> sleeve_service_;
  std::shared_ptr<ImagePoolService>  image_pool_service_;
  std::shared_ptr<PipelineMgmtService>   pipeline_service_;

  std::shared_ptr<PipelineScheduler> pipeline_scheduler_;

  ThreadPool                         export_thread_pool_{4};

  std::list<ExportTask>              export_queue_;
  std::mutex                         queue_mutex_;

  // We only use one mutex to protect the result collection
  // So we only support one export session involving multiple exports at a time
  std::mutex                         result_mutex_;

  void RunExportRenderTask(const ExportTask& task);

 public:
  ExportService() = delete;
  ExportService(std::shared_ptr<SleeveServiceImpl> sleeve_service,
                std::shared_ptr<ImagePoolService>  image_pool_service,
                std::shared_ptr<PipelineMgmtService>   pipeline_service)
      : sleeve_service_(std::move(sleeve_service)),
        image_pool_service_(std::move(image_pool_service)),
        pipeline_service_(std::move(pipeline_service)) {
    pipeline_scheduler_ = RenderService::GetThumbnailOrExportScheduler(8);
  };

  ExportService(const ExportService&)            = delete;
  ExportService& operator=(const ExportService&) = delete;

  auto           EnqueueExportTask(const ExportTask& task) -> void {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    export_queue_.emplace_back(task);
  }
  void RemoveExportTask(sl_element_id_t sleeve_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    export_queue_.remove_if(
        [sleeve_id](const ExportTask& task) { return task.sleeve_id_ == sleeve_id; });
  };
  void ClearAllExportTasks() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    export_queue_.clear();
  };

  void ExportAll(std::function<void(std::shared_ptr<std::vector<ExportResult>>)> callback);
};
};  // namespace puerhlab