//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <memory>

#include "edit/operators/op_kernel.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "image/image.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

namespace alcedo {
enum class RenderType { FAST_PREVIEW, FULL_RES_PREVIEW, FULL_RES_EXPORT, THUMBNAIL };

struct RenderDesc {
  RenderType render_type_  = RenderType::FAST_PREVIEW;

  int        x_            = 0;
  int        y_            = 0;
  float      scale_factor_x_ = 1.0f;
  float      scale_factor_y_ = 1.0f;
  bool       use_viewport_region_ = true;
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
  std::shared_ptr<CPUPipelineExecutor>                        pipeline_executor_;
  std::shared_ptr<ImageBuffer>                                input_;
  std::shared_ptr<Image>                                      input_desc_;

  std::shared_ptr<std::promise<std::shared_ptr<ImageBuffer>>> result_;    // used for blocking tasks
  std::optional<std::function<void(ImageBuffer&)>>            callback_;  // used for callback tasks
  std::optional<std::function<void(ImageBuffer&, uint32_t)>>
              seq_callback_;  // used for callback tasks

  TaskOptions options_;

  void        SetExecutorRenderParams();
  void        ResetPreviewRenderParams();
  void        ResetThumbnailRenderParams();
};
};  // namespace alcedo
