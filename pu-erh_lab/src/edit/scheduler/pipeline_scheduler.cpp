#include "edit/scheduler/pipeline_scheduler.hpp"

#include <mutex>

namespace puerhlab {
PipelineScheduler::PipelineScheduler() : _thread_pool(1) {}

void PipelineScheduler::ScheduleTask(PipelineTask&& task) {
  task._task_id = _id_generator.GenerateID();
  _thread_pool.Submit([task = std::move(task), &lock = _scheduler_lock]() mutable {
    std::unique_lock<std::mutex> guard(lock);
    using clock = std::chrono::high_resolution_clock;
    auto result = task._pipeline_executor->Apply(task._input);
    auto output = result->GetCPUData().clone();
    guard.unlock();

    if (task._options._is_callback && task._callback) {
      (*task._callback)(*result);
    }
    if (task._options._is_seq_callback && task._seq_callback) {
      (*task._seq_callback)(*result, task._task_id);
    }
    if (task._options._is_blocking) {
      task._result->set_value(result);  // Notify the waiting thread
    }
  });
}
}  // namespace puerhlab