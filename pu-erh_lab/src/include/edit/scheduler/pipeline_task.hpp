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
#include <functional>
#include <future>
#include <memory>

#include "edit/pipeline/pipeline.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct TaskOptions {
  bool          _is_preview;  // if true, use preview pipeline

  bool          _is_blocking;      // if true, wait for the task to finish
  bool          _is_callback;      // if true, use callback to return result
  bool          _is_seq_callback;  // if true, use sequential callback to return result
  PriorityLevel _task_priority;    // task priority level, not used yet
};
struct PipelineTask {
  uint32_t                                                    _task_id;
  std::shared_ptr<PipelineExecutor>                           _pipeline_executor;
  std::shared_ptr<ImageBuffer>                                _input;

  std::shared_ptr<std::promise<std::shared_ptr<ImageBuffer>>> _result;    // used for blocking tasks
  std::optional<std::function<void(ImageBuffer&)>>            _callback;  // used for callback tasks
  std::optional<std::function<void(ImageBuffer&, uint32_t)>>
              _seq_callback;  // used for callback tasks

  TaskOptions _options;
};
};  // namespace puerhlab