#pragma once

#include "concurrency/thread_pool.hpp"
#include "pipeline_task.hpp"

namespace puerhlab {
class PipelineScheduler {
 private:
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