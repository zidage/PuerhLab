//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>

#include "edit/operators/op_base.hpp"

namespace puerhlab {
class ImageBuffer;
class IFrameSink;

class GPUPipelineImpl {
 public:
  virtual ~GPUPipelineImpl() = default;

  virtual void SetInputImage(std::shared_ptr<ImageBuffer> input_image) = 0;
  virtual void SetParams(OperatorParams& params)                        = 0;
  virtual void SetFrameSink(IFrameSink* frame_sink)                    = 0;
  virtual void Execute(std::shared_ptr<ImageBuffer> output)            = 0;
  virtual void ReleaseResources()                                      = 0;
};

class GPUPipelineWrapper {
 public:
  GPUPipelineWrapper();
  ~GPUPipelineWrapper();

  void SetInputImage(std::shared_ptr<ImageBuffer> input_image);

  void SetParams(OperatorParams& params);

  void SetFrameSink(IFrameSink* frame_sink);

  void Execute(std::shared_ptr<ImageBuffer> output);

  [[nodiscard]] auto HasAcceleratedBackend() const -> bool;

  // Frees persistent GPU allocations used by the pipeline (scratch buffers, LUTs, etc.).
  // Intended for batch export to avoid holding large VRAM allocations across many pipelines.
  void ReleaseResources();

 private:
  std::unique_ptr<GPUPipelineImpl> impl_;
};
}  // namespace puerhlab
