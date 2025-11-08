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
  ThreadPool _thread_pool;  // use thred pool for now, can be changed to task scheduler later

  std::mutex _scheduler_lock;

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