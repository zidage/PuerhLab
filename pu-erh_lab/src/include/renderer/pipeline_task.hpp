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

#include "edit/operators/op_kernel.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class RenderType { FAST_PREVIEW, FULL_RES_PREVIEW, FULL_RES_EXPORT, THUMBNAIL };

struct RenderDesc {
  RenderType render_type_  = RenderType::FAST_PREVIEW;

  int        x_            = 0;
  int        y_            = 0;
  float      scale_factor_ = 1.0f;
};

struct TaskOptions {
  RenderDesc    render_desc_;

  bool          is_blocking_;      // if true, wait for the task to finish
  bool          is_callback_;      // if true, use callback to return result
  bool          is_seq_callback_;  // if true, use sequential callback to return result
  PriorityLevel task_priority_;    // task priority level, not used yet
};
struct PipelineTask {
  uint32_t                                                    task_id_;
  std::shared_ptr<PipelineExecutor>                           pipeline_executor_;
  std::shared_ptr<ImageBuffer>                                input_;

  std::shared_ptr<std::promise<std::shared_ptr<ImageBuffer>>> result_;    // used for blocking tasks
  std::optional<std::function<void(ImageBuffer&)>>            callback_;  // used for callback tasks
  std::optional<std::function<void(ImageBuffer&, uint32_t)>>
              seq_callback_;  // used for callback tasks

  TaskOptions options_;

  void        SetExecutorRenderParams();
};
};  // namespace puerhlab