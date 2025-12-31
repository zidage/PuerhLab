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

#include <cstdint>
#include <mutex>

#include "concurrency/thread_pool.hpp"
#include "pipeline_task.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
class PipelineScheduler {
 private:
  IncrID::IDGenerator<uint32_t> _id_generator{0};

  std::mutex                    _scheduler_lock;
  ThreadPool _thread_pool;  // use thred pool for now, can be changed to task scheduler later

 public:
  explicit PipelineScheduler();

  /**
   * @brief Schedule a pipeline task
   *
   * @param task
   */
  void ScheduleTask(PipelineTask&& task);
};
};  // namespace puerhlab