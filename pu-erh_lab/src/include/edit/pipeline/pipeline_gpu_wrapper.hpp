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