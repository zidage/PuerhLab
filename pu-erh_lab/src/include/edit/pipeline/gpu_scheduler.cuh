#pragma once

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/hal/interface.h>

#include <cstddef>
#include <memory>
#include <opencv2/core/cuda.hpp>

#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "kernel_stream_gpu.cuh"

namespace puerhlab {
namespace CUDA {
template <typename KernelStreamT>
class GPU_KernelLauncher {
 private:
  std::shared_ptr<ImageBuffer> _input_img;

  float4*                      _work_buffer    = nullptr;
  float4*                      _temp_buffer    = nullptr;
  size_t                       _allocated_size = 0;

  std::shared_ptr<ImageBuffer> _output_img;

  KernelStreamT                _kernel_stream;

  GPUOperatorParams            _params;

 public:
  GPU_KernelLauncher(std::shared_ptr<ImageBuffer> input_img, KernelStreamT kernel_stream)
      : _input_img(input_img), _kernel_stream(kernel_stream) {}

  ~GPU_KernelLauncher() {
    if (_work_buffer) {
      cudaFree(_work_buffer);
      _work_buffer = nullptr;
    }
    if (_temp_buffer) {
      cudaFree(_temp_buffer);
      _temp_buffer = nullptr;
    }

    _params.to_ws_lut.Reset();
    _params.lmt_lut.Reset();
    _params.to_output_lut.Reset();
  }

  void SetInputImage(std::shared_ptr<ImageBuffer> input_img) {
    _input_img = input_img;
    _input_img->SyncToGPU();

    cv::cuda::GpuMat gpu_mat     = _input_img->GetGPUData();

    size_t           width       = gpu_mat.cols;
    size_t           height      = gpu_mat.rows;
    size_t           needed_size = width * height * sizeof(float4);

    if (needed_size > _allocated_size) {
      if (_work_buffer) {
        cudaFree(_work_buffer);
        _work_buffer = nullptr;
      }
      if (_temp_buffer) {
        cudaFree(_temp_buffer);
        _temp_buffer = nullptr;
      }

      cudaMalloc((void**)&_work_buffer, needed_size);
      cudaMalloc((void**)&_temp_buffer, needed_size);
      _allocated_size = needed_size;
    }
  }

  void SetOutputImage(std::shared_ptr<ImageBuffer> output_img) { _output_img = output_img; }

  void SetParams(GPUOperatorParams& params) {
    _params = params;  // copy the params into the scheduler
  }

  void SetParams(OperatorParams& cpu_params) {
    _params = GPUParamsConverter::ConvertFromCPU(cpu_params);
  }

  void Execute() {
    if (!_input_img || !_work_buffer) {
      throw std::runtime_error("Input image not set or work buffer not allocated.");
    }

    cv::cuda::GpuMat gpu_mat = _input_img->GetGPUData();
    size_t           width   = gpu_mat.cols;
    size_t           height  = gpu_mat.rows;
    cudaMemcpy2D(_work_buffer, width * sizeof(float4), gpu_mat.ptr<float4>(), gpu_mat.step,
                 width * sizeof(float4), height, cudaMemcpyDeviceToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float4* result_ptr = _kernel_stream.Process(_work_buffer, _temp_buffer, static_cast<int>(width),
                                                static_cast<int>(height),
                                                static_cast<size_t>(width), _params, stream);
    // Process() will synchronize the stream internally
    cudaStreamDestroy(stream);

    if (_output_img) {
      _output_img->InitGPUData(width, height, CV_32FC4);
      cv::cuda::GpuMat output_gpu_mat = _output_img->GetGPUData();
      cudaMemcpy2D(output_gpu_mat.ptr<float4>(), output_gpu_mat.step, result_ptr,
                   width * sizeof(float4), width * sizeof(float4), height,
                   cudaMemcpyDeviceToDevice);
      _output_img->SetGPUDataValid(true);
    }
  }
};
}  // namespace CUDA
};  // namespace puerhlab