#pragma once

#include <functional>
#include <future>
#include <memory>

#include "edit/pipeline/pipeline.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct TaskOptions {
  bool          _is_blocking;    // if true, wait for the task to finish
  bool          _is_callback;    // if true, use callback to return result
  PriorityLevel _task_priority;  // task priority level, not used yet
};
struct PipelineTask {
  std::shared_ptr<PipelineExecutor>                _pipeline_executor;
  std::shared_ptr<ImageBuffer>                     _input;

  std::shared_ptr<std::promise<ImageBuffer>>       _result;    // used for blocking tasks
  std::optional<std::function<void(ImageBuffer&)>> _callback;  // used for callback tasks

  TaskOptions                                      _options;
};
};  // namespace puerhlab