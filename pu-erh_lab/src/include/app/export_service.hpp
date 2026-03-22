//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

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
  bool        wrote_ultra_hdr_ = false;
  bool        used_embedded_profile_fallback_ = false;
  std::string message_;
};

struct ExportProgress {
  size_t total_     = 0;
  size_t completed_ = 0;
  size_t succeeded_ = 0;
  size_t failed_    = 0;
};

class ExportService {
 private:
  std::shared_ptr<SleeveServiceImpl> sleeve_service_;
  std::shared_ptr<ImagePoolService>  image_pool_service_;
  std::shared_ptr<PipelineMgmtService>   pipeline_service_;

  std::shared_ptr<PipelineScheduler> pipeline_scheduler_;

  std::list<ExportTask>              export_queue_;
  std::mutex                         queue_mutex_;

  // We only use one mutex to protect the result collection
  // So we only support one export session involving multiple exports at a time
  std::mutex                         result_mutex_;

  // Keep this last so worker threads are joined before other members are torn down.
  ThreadPool                         export_thread_pool_{4};

  auto RunExportRenderTask(const ExportTask& task) -> ExportResult;

 public:
  ExportService() = delete;
  ExportService(std::shared_ptr<SleeveServiceImpl> sleeve_service,
                std::shared_ptr<ImagePoolService>  image_pool_service,
                std::shared_ptr<PipelineMgmtService>   pipeline_service)
      : sleeve_service_(std::move(sleeve_service)),
        image_pool_service_(std::move(image_pool_service)),
        pipeline_service_(std::move(pipeline_service)) {
    pipeline_scheduler_ = RenderService::GetThumbnailOrExportScheduler(1);
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
  void ExportAll(std::function<void(const ExportProgress&)> progress_callback,
                 std::function<void(std::shared_ptr<std::vector<ExportResult>>)> callback);
};
};  // namespace puerhlab
