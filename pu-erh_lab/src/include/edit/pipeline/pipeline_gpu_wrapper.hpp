//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>

#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {
class GPUPipelineImpl;

class GPUPipelineWrapper {
 public:
  GPUPipelineWrapper();
  ~GPUPipelineWrapper();

  void SetInputImage(std::shared_ptr<ImageBuffer> input_image);

  void SetParams(OperatorParams& params);

  void SetFrameSink(IFrameSink* frame_sink);

  void Execute(std::shared_ptr<ImageBuffer> output);

  // Frees persistent GPU allocations used by the pipeline (scratch buffers, LUTs, etc.).
  // Intended for batch export to avoid holding large VRAM allocations across many pipelines.
  void ReleaseResources();

 private:
  std::unique_ptr<GPUPipelineImpl> impl_;


};
};  // namespace puerhlab
