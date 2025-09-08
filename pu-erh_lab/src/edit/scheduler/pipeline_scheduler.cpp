#include "edit/scheduler/pipeline_scheduler.hpp"

namespace puerhlab {
PipelineScheduler::PipelineScheduler() : _thread_pool(8) {}

void PipelineScheduler::ScheduleTask(PipelineTask&& task) {
  _thread_pool.Submit([task = std::move(task)]() mutable {
    ImageBuffer result = task._pipeline_executor->Apply(*task._input);
    if (task._options._is_blocking) {
      task._result->set_value(std::move(result));
    }
    if (task._options._is_callback && task._callback) {
      (*task._callback)(result);
    }
  });
}
}  // namespace puerhlab