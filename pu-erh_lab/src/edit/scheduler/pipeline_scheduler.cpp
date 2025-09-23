#include "edit/scheduler/pipeline_scheduler.hpp"

namespace puerhlab {
PipelineScheduler::PipelineScheduler() : _thread_pool(8) {}

void PipelineScheduler::ScheduleTask(PipelineTask&& task) {
  task._task_id = _id_generator.GenerateID();
  _thread_pool.Submit([task = std::move(task)]() mutable {
    auto result = task._pipeline_executor->Apply(task._input);

    if (task._options._is_callback && task._callback) {
      (*task._callback)(*result);
    }
    if (task._options._is_seq_callback && task._seq_callback) {
      (*task._seq_callback)(*result, task._task_id);
    }
    if (task._options._is_blocking) {
      task._result->set_value({std::move(*result)});  // Notify the waiting thread
    }
  });
}
}  // namespace puerhlab