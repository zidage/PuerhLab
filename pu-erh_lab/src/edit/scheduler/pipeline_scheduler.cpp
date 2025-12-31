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
PipelineScheduler::PipelineScheduler() : _thread_pool(1) {}

void PipelineScheduler::ScheduleTask(PipelineTask&& task) {
  task._task_id = _id_generator.GenerateID();
  _thread_pool.Submit([task = std::move(task), &lock = _scheduler_lock]() mutable {
    std::shared_ptr<ImageBuffer> result_copy;
    {
      std::unique_lock<std::mutex> guard(lock);
      if (task._input) {
        auto result = task._pipeline_executor->Apply(task._input);
        if (result) {
          result_copy = std::make_shared<ImageBuffer>(result->GetCPUData());
        }
      }
    }

    if (result_copy) {
      if (task._options._is_callback && task._callback) {
        (*task._callback)(*result_copy);
      }
      if (task._options._is_seq_callback && task._seq_callback) {
        (*task._seq_callback)(*result_copy, task._task_id);
      }
      if (task._options._is_blocking) {
        task._result->set_value(result_copy);  // Notify the waiting thread
      }
    } else if (task._options._is_blocking) {
      // In case of failure, set nullptr
      task._result->set_value(nullptr);
    }
  });
}
}  // namespace puerhlab