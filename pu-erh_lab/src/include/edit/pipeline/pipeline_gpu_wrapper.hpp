#pragma once

#include <memory>

#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
class GPUPipelineImpl;

class GPUPipelineWrapper {
 public:
  GPUPipelineWrapper();
  ~GPUPipelineWrapper();

  void SetInputImage(std::shared_ptr<ImageBuffer> input_image);

  void SetParams(OperatorParams& params);

  void Execute(std::shared_ptr<ImageBuffer> output);

 private:
  std::unique_ptr<GPUPipelineImpl> _impl;


};
};  // namespace puerhlab