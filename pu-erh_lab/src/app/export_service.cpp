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

#include "app/export_service.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>

#include "image/image.hpp"
#include "image/image_buffer.hpp"
#include "io/image/image_loader.hpp"
#include "io/image/image_writer.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "type/type.hpp"

namespace puerhlab {
void ExportService::RunExportRenderTask(const ExportTask& task) {
  // Get the pipeline executor from sleeve service
  auto pipeline_guard = pipeline_service_->LoadPipeline(task.sleeve_id_);
  if (!pipeline_guard || !pipeline_guard->pipeline_) {
    throw std::runtime_error("[ERROR] ExportService: Failed to load pipeline for sleeve id " +
                             std::to_string(task.sleeve_id_));
  }
  // Get the image from image pool service
  auto img_src_path = image_pool_service_->Read<std::filesystem::path>(
      task.image_id_, [](std::shared_ptr<Image> img) { return img->image_path_; });

  // Create a pipeline task for export
  PipelineTask render_task;
  // To avoid reading too many images into memory at once, we let the pipeline load the image
  // So we create a dummy Image object with only the path set
  render_task.input_desc_ =
      std::make_shared<Image>(img_src_path, ImageType::DEFAULT);
  render_task.pipeline_executor_                 = pipeline_guard->pipeline_;
  render_task.options_.is_blocking_              = true;
  render_task.options_.is_callback_              = false;
  // Use full res export, even though the task requires resizing,
  // to benefit from the super sampling
  render_task.options_.render_desc_.render_type_ = RenderType::FULL_RES_EXPORT;
  // Set export options in the pipeline executor
  auto render_promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
  render_task.result_ = render_promise;
  auto render_future  = render_promise->get_future();
  // Schedule the render task
  pipeline_scheduler_->ScheduleTask(std::move(render_task));
  // Wait for the render to complete
  auto rendered_image = render_future.get();
  // Save pipeline back to storage
  pipeline_service_->SavePipeline(pipeline_guard);
  // Use ImageWriter to write the image to disk
  ImageWriter::WriteImageToPath(
      img_src_path, rendered_image,
      task.options_);
}

void ExportService::ExportAll(
    std::function<void(std::shared_ptr<std::vector<ExportResult>>)> callback) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  auto                        results = std::make_shared<std::vector<ExportResult>>();

  size_t queue_size = export_queue_.size();
  while (!export_queue_.empty()) {
    ExportTask task = export_queue_.front();
    export_queue_.pop_front();

    // Export in thread pool
    export_thread_pool_.Submit([this, task, results, callback, queue_size]() {
      ExportResult result;
      // Do export, this call will block until done
      try {
        RunExportRenderTask(task);
        result.success_ = true;
      } catch (const std::exception& e) {
        result.success_ = false;
        result.message_ = e.what();
      }

      // Store result
      {
        std::lock_guard<std::mutex> res_lock(result_mutex_);
        results->push_back(result);
      }

      // If all done, call the callback
      {
        std::lock_guard<std::mutex> res_lock(result_mutex_);
        if (results->size() == queue_size) {
          callback(results);
        }
      }
    });
  }
};
};  // namespace puerhlab
