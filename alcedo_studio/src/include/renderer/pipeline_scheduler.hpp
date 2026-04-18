//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <mutex>

#include "concurrency/thread_pool.hpp"
#include "pipeline_task.hpp"
#include "utils/id/id_generator.hpp"

namespace alcedo {
class PipelineScheduler {
 private:
  IncrID::IDGenerator<uint32_t> id_generator_{0};

  std::mutex                    scheduler_lock_;
  ThreadPool thread_pool_;  // use thred pool for now, can be changed to task scheduler later

  
 public:
  explicit PipelineScheduler();
  explicit PipelineScheduler(size_t thread_count);

  /**
   * @brief Schedule a pipeline task
   *
   * @param task
   */
  void ScheduleTask(PipelineTask&& task);
};
};  // namespace alcedo