//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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
namespace {

auto ResolveExportColorProfileConfig(const OperatorParams& params) -> ExportColorProfileConfig {
  return ExportColorProfileConfig{params.to_output_params_.encoding_space_,
                                  params.to_output_params_.eotf_,
                                  params.to_output_params_.peak_luminance_};
}

}  // namespace

auto ExportService::RunExportRenderTask(const ExportTask& task) -> ExportResult {
  ExportResult result;

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

  // Inject pre-extracted raw metadata from the real Image into the pipeline
  // so downstream operators resolve eagerly.
  try {
    auto full_img = image_pool_service_->Read<std::shared_ptr<Image>>(
        task.image_id_, [](const std::shared_ptr<Image>& i) { return i; });
    if (full_img && full_img->HasRawColorContext()) {
      pipeline_guard->pipeline_->InjectRawMetadata(full_img->GetRawColorContext());
    }
  } catch (...) {
    // Non-fatal: metadata injection is best-effort.
  }

  // Use full res export, even though the task requires resizing,
  // to benefit from the super sampling
  render_task.options_.render_desc_.render_type_ = RenderType::FULL_RES_EXPORT;
  // Set export options in the pipeline executor
  auto render_promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
  render_task.result_ = render_promise;
  auto render_future  = render_promise->get_future();
  // Schedule the render task
  pipeline_scheduler_->ScheduleTask(std::move(render_task));

  std::shared_ptr<ImageBuffer> rendered_image;
  try {
    // Wait for the render to complete
    rendered_image = render_future.get();
  } catch (...) {
    pipeline_service_->SavePipeline(pipeline_guard);
    throw;
  }

  // Save pipeline back to storage
  const auto export_profile =
      ResolveExportColorProfileConfig(pipeline_guard->pipeline_->GetGlobalParams());
  const bool wrote_ultra_hdr = ImageWriter::ShouldWriteUltraHdr(task.options_, export_profile);
  const bool requested_ultra_hdr =
      task.options_.hdr_export_mode_ == ExportFormatOptions::HDR_EXPORT_MODE::ULTRA_HDR &&
      task.options_.format_ == ImageFormatType::JPEG;
  pipeline_service_->SavePipeline(pipeline_guard);
  // Use ImageWriter to write the image to disk
  ImageWriter::WriteImageToPath(img_src_path, rendered_image, task.options_, export_profile);
  result.success_ = true;
  result.wrote_ultra_hdr_ = wrote_ultra_hdr;
  result.used_embedded_profile_fallback_ = requested_ultra_hdr && !wrote_ultra_hdr;
  return result;
}

void ExportService::ExportAll(
    std::function<void(std::shared_ptr<std::vector<ExportResult>>)> callback) {
  ExportAll({}, std::move(callback));
}

void ExportService::ExportAll(
    std::function<void(const ExportProgress&)>                    progress_callback,
    std::function<void(std::shared_ptr<std::vector<ExportResult>>)> callback) {
  auto results = std::make_shared<std::vector<ExportResult>>();
  std::vector<ExportTask> tasks;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    tasks.reserve(export_queue_.size());
    while (!export_queue_.empty()) {
      tasks.push_back(export_queue_.front());
      export_queue_.pop_front();
    }
  }

  const size_t queue_size = tasks.size();
  if (queue_size == 0) {
    try {
      callback(results);
    } catch (...) {
    }
    return;
  }

  auto completed = std::make_shared<std::atomic_size_t>(0);
  auto succeeded = std::make_shared<std::atomic_size_t>(0);
  auto failed    = std::make_shared<std::atomic_size_t>(0);
  for (const auto& task : tasks) {
    // Export in thread pool
    export_thread_pool_.Submit([this, task, results, progress_callback, callback, completed,
                                succeeded, failed, queue_size]() {
      ExportResult result;
      // Do export, this call will block until done
      try {
        result = RunExportRenderTask(task);
      } catch (const std::exception& e) {
        result.success_ = false;
        result.message_ = e.what();
      } catch (...) {
        result.success_ = false;
        result.message_ = "Unknown export error";
      }

      const bool export_ok = result.success_;

      // Store result
      {
        std::lock_guard<std::mutex> res_lock(result_mutex_);
        results->push_back(std::move(result));
      }

      if (export_ok) {
        succeeded->fetch_add(1, std::memory_order_acq_rel);
      } else {
        failed->fetch_add(1, std::memory_order_acq_rel);
      }

      // If all done, call the callback
      const size_t finished = completed->fetch_add(1, std::memory_order_acq_rel) + 1;
      if (progress_callback) {
        try {
          progress_callback(ExportProgress{
              .total_     = queue_size,
              .completed_ = finished,
              .succeeded_ = succeeded->load(std::memory_order_acquire),
              .failed_    = failed->load(std::memory_order_acquire),
          });
        } catch (...) {
        }
      }
      if (finished == queue_size) {
        try {
          callback(results);
        } catch (...) {
        }
      }
    });
  }
};
};  // namespace puerhlab
