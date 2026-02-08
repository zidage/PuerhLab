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

#include "renderer/pipeline_scheduler.hpp"

#include <easy/profiler.h>

#include <exception>
#include <memory>
#include <mutex>

#include "image/image_buffer.hpp"
#include "io/image/image_loader.hpp"
#include "renderer/pipeline_task.hpp"

namespace puerhlab {
void PipelineTask::SetExecutorRenderParams() {
  if (!pipeline_executor_) {
    return;
  }
  auto& desc = options_.render_desc_;
  pipeline_executor_->SetRenderRegion(desc.x_, desc.y_, desc.scale_factor_);
  if (desc.render_type_ == RenderType::FAST_PREVIEW) {
    pipeline_executor_->SetForceCPUOutput(false);
    pipeline_executor_->SetRenderRes(false, 4096);
    pipeline_executor_->SetEnableCache(true);
    // The default decode res is full, this call will be effective only when changed before
    pipeline_executor_->SetDecodeRes(DecodeRes::FULL);
    return;
  }
  if (desc.render_type_ == RenderType::THUMBNAIL) {
    pipeline_executor_->SetForceCPUOutput(true);
    pipeline_executor_->SetRenderRes(false, 1024);
    pipeline_executor_->SetEnableCache(false);
    pipeline_executor_->SetDecodeRes(DecodeRes::QUARTER);
    return;
  }
  if (desc.render_type_ == RenderType::FULL_RES_PREVIEW) {
    pipeline_executor_->SetRenderRes(true);
    pipeline_executor_->SetForceCPUOutput(false);
    // pipeline_executor_->SetEnableCache(true);
    // pipeline_executor_->SetDecodeRes(DecodeRes::FULL);
    return;
  }
  if (desc.render_type_ == RenderType::FULL_RES_EXPORT) {
    pipeline_executor_->SetRenderRes(true);
    pipeline_executor_->SetForceCPUOutput(true);
    pipeline_executor_->SetEnableCache(false);
    // The default decode res is full, this call will be effective only when changed before
    pipeline_executor_->SetDecodeRes(DecodeRes::FULL);
    return;
  }
  throw std::runtime_error("[ERROR] PipelineTask: Unknown render type");
}

void PipelineTask::ResetPreviewRenderParams() {
  if (!pipeline_executor_) {
    return;
  }
  // A simple status machine to automatically set back to fast preview mode
  pipeline_executor_->SetRenderRes(false, 4096);
  pipeline_executor_->SetForceCPUOutput(false);
  // pipeline_executor_->SetEnableCache(true);
  // pipeline_executor_->SetDecodeRes(DecodeRes::FULL);
}

void PipelineTask::ResetThumbnailRenderParams() {
  if (!pipeline_executor_) {
    return;
  }
  // To make sure thumbnail mode is idempotent
  pipeline_executor_->SetRenderRes(true, 4096);
  pipeline_executor_->SetForceCPUOutput(false);
  pipeline_executor_->SetEnableCache(true);
  pipeline_executor_->SetDecodeRes(DecodeRes::FULL);
}

PipelineScheduler::PipelineScheduler() : thread_pool_(1) {}

PipelineScheduler::PipelineScheduler(size_t thread_count) : thread_pool_(thread_count) {}

void PipelineScheduler::ScheduleTask(PipelineTask&& task) {
  task.task_id_ = id_generator_.GenerateID();
  thread_pool_.Submit([task = std::move(task)]() mutable {
    const auto set_blocking_value = [&task](std::shared_ptr<ImageBuffer> value) {
      if (!task.options_.is_blocking_ || !task.result_) {
        return;
      }
      try {
        task.result_->set_value(std::move(value));
      } catch (...) {
      }
    };

    const auto set_blocking_exception = [&task]() {
      if (!task.options_.is_blocking_ || !task.result_) {
        return;
      }
      try {
        task.result_->set_exception(std::current_exception());
      } catch (...) {
      }
    };

    try {
      std::shared_ptr<ImageBuffer> result_copy;
      {
        if (task.input_desc_ && !task.input_) {
          // Load image data into buffer
          task.input_ = std::make_shared<ImageBuffer>(
              ByteBufferLoader::LoadByteBufferFromImage(task.input_desc_));
        }
        if (task.input_) {
          std::unique_lock<std::mutex> render_lock;
          if (task.pipeline_executor_) {
            render_lock = std::unique_lock<std::mutex>(task.pipeline_executor_->GetRenderLock());
          }

          auto& render_desc = task.options_.render_desc_;
          if (!(render_desc.render_type_ == RenderType::FAST_PREVIEW)) {
            task.SetExecutorRenderParams();
          }

          auto result = task.pipeline_executor_->Apply(task.input_);

          if (render_desc.render_type_ == RenderType::FAST_PREVIEW ||
              render_desc.render_type_ == RenderType::FULL_RES_PREVIEW || !result ||
              !result->gpu_data_valid_ || result->GetCPUData().empty()) {
            set_blocking_value(result);

            if (render_desc.render_type_ == RenderType::FULL_RES_PREVIEW) {
              // Reset to fast preview mode after full res preview
              task.ResetPreviewRenderParams();
            }

            if (render_desc.render_type_ == RenderType::THUMBNAIL) {
              // Reset to FULL_RES_PREVIEW mode after thumbnail render
              task.ResetThumbnailRenderParams();
              // task.pipeline_executor_->ClearAllIntermediateBuffers();
            }

            if (render_desc.render_type_ == RenderType::FULL_RES_EXPORT) {
              // Reset to fast preview mode after full res export so the next
              // FAST_PREVIEW task (which skips SetExecutorRenderParams) does not
              // inherit export-mode flags (force_cpu_output, disabled cache, full res).
              task.ResetPreviewRenderParams();
              task.pipeline_executor_->SetEnableCache(true);
            }
            return;
          }

          result_copy = std::make_shared<ImageBuffer>(result->GetCPUData());
        }
      }

      if (result_copy) {
        if (task.options_.is_callback_ && task.callback_) {
          (*task.callback_)(*result_copy);
        }
        if (task.options_.is_seq_callback_ && task.seq_callback_) {
          (*task.seq_callback_)(*result_copy, task.task_id_);
        }
        set_blocking_value(result_copy);

        // Cleanup after callback completes
        auto& render_desc = task.options_.render_desc_;
        if (render_desc.render_type_ == RenderType::THUMBNAIL ||
            render_desc.render_type_ == RenderType::FULL_RES_EXPORT) {
          if (render_desc.render_type_ == RenderType::THUMBNAIL) {
            // Reset to fast preview mode after thumbnail render
            task.ResetPreviewRenderParams();
          }
          if (render_desc.render_type_ == RenderType::FULL_RES_EXPORT) {
            // Reset to fast preview mode after full res export
            task.ResetPreviewRenderParams();
          }
          // Release all intermediate buffers to free memory
          // if (render_desc.render_type_ == RenderType::FULL_RES_EXPORT) {
          //   task.pipeline_executor_->ReleaseAllGPUResources();
          // }
        }
      } else {
        // In case of failure, set nullptr
        set_blocking_value(nullptr);
      }
    } catch (...) {
      set_blocking_exception();
    }
  });
}
}  // namespace puerhlab
