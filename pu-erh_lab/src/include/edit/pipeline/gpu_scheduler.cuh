#pragma once

#include <memory>

#include "image/image_buffer.hpp"
#include "kernel_stream_gpu.cuh"


namespace puerhlab {
namespace CUDA {
template <typename KernelStreamT>
class GPUScheduler {
 private:
  std::shared_ptr<ImageBuffer> _input_img;

  KernelStreamT                _kernel_stream;

 public:
  GPUScheduler(std::shared_ptr<ImageBuffer> input_img, KernelStreamT kernel_stream)
      : _input_img(input_img), _kernel_stream(kernel_stream) {}

  void SetInputImage(std::shared_ptr<ImageBuffer> input_img) {
    _input_img = input_img;
  }
  
  void Execute(std::shared_ptr<ImageBuffer> output_img) {
    
  }
};
}  // namespace CUDA
};  // namespace puerhlab