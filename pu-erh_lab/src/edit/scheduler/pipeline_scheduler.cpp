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

#include "edit/scheduler/pipeline_scheduler.hpp"

#include <memory>
#include <mutex>

#include "image/image_buffer.hpp"

namespace puerhlab {
PipelineScheduler::PipelineScheduler() : thread_pool_(1) {}

void PipelineScheduler::ScheduleTask(PipelineTask&& task) {
  task.task_id_ = id_generator_.GenerateID();
  thread_pool_.Submit([task = std::move(task), &lock = scheduler_lock_]() mutable {
    std::shared_ptr<ImageBuffer> result_copy;
    {
      std::unique_lock<std::mutex> guard(lock);
      if (task.input_) {
        auto result = task.pipeline_executor_->Apply(task.input_);
        if (result) {
          result_copy = std::make_shared<ImageBuffer>(result->GetCPUData());
        }
      }
    }

    if (result_copy) {
      if (task.options_.is_callback_ && task.callback_) {
        (*task.callback_)(*result_copy);
      }
      if (task.options_.is_seq_callback_ && task.seq_callback_) {
        (*task.seq_callback_)(*result_copy, task.task_id_);
      }
      if (task.options_.is_blocking_) {
        task.result_->set_value(result_copy);  // Notify the waiting thread
      }
    } else if (task.options_.is_blocking_) {
      // In case of failure, set nullptr
      task.result_->set_value(nullptr);
    }
  });
}
}  // namespace puerhlab