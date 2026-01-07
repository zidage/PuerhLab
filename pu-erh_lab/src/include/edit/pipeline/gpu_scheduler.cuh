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

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/hal/interface.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <opencv2/core/cuda.hpp>

#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "kernel_stream_gpu.cuh"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {
namespace CUDA {
template <typename KernelStreamT>
class GPU_KernelLauncher {
 private:
  std::shared_ptr<ImageBuffer> input_img_;

  float4*                      work_buffer_    = nullptr;
  float4*                      temp_buffer_    = nullptr;
  size_t                       allocated_size_ = 0;

  std::shared_ptr<ImageBuffer> output_img_;

  KernelStreamT                kernel_stream_;

  GPUOperatorParams            params_;

  IFrameSink*                  frame_sink_ = nullptr;

 public:
  GPU_KernelLauncher(std::shared_ptr<ImageBuffer> input_img, KernelStreamT kernel_stream)
      : input_img_(input_img), kernel_stream_(kernel_stream) {}

  ~GPU_KernelLauncher() {
    if (work_buffer_) {
      cudaFree(work_buffer_);
      work_buffer_ = nullptr;
    }
    if (temp_buffer_) {
      cudaFree(temp_buffer_);
      temp_buffer_ = nullptr;
    }

    params_.to_ws_lut_.Reset();
    params_.lmt_lut_.Reset();
    params_.to_output_lut_.Reset();
    params_.to_output_params_.odt_params_.Reset();
  }

  void SetInputImage(std::shared_ptr<ImageBuffer> input_img) {
    input_img_ = input_img;
    input_img_->SyncToGPU();
    cv::cuda::GpuMat gpu_mat     = input_img_->GetGPUData();

    size_t           width       = gpu_mat.cols;
    size_t           height      = gpu_mat.rows;
    size_t           needed_size = width * height * sizeof(float4);

    if (needed_size > allocated_size_) {
      if (work_buffer_) {
        cudaFree(work_buffer_);
        work_buffer_ = nullptr;
      }

      cudaMalloc((void**)&work_buffer_, needed_size);
      cudaMalloc((void**)&temp_buffer_, needed_size);
      allocated_size_ = needed_size;
    }
  }

  void SetOutputImage(std::shared_ptr<ImageBuffer> output_img) { output_img_ = output_img; }

  void SetParams(OperatorParams& cpu_params) {
    params_ = GPUParamsConverter::ConvertFromCPU(cpu_params, params_);
  }

  void SetFrameSink(IFrameSink* frame_sink) { frame_sink_ = frame_sink; }

  void Execute() {
    if (!input_img_ || !work_buffer_) {
      throw std::runtime_error("Input image not set or work buffer not allocated.");
    }

    cv::cuda::GpuMat gpu_mat = input_img_->GetGPUData();
    size_t           width   = gpu_mat.cols;
    size_t           height  = gpu_mat.rows;
    {
      const auto copy_err = cudaMemcpy2D(work_buffer_, width * sizeof(float4), gpu_mat.ptr<float4>(),
                                         gpu_mat.step, width * sizeof(float4), height,
                                         cudaMemcpyDeviceToDevice);
      if (copy_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy2D (input->work) failed: ") +
                                 cudaGetErrorString(copy_err));
      }
    }

    cudaStream_t stream;
    {
      const auto stream_err = cudaStreamCreate(&stream);
      if (stream_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                                 cudaGetErrorString(stream_err));
      }
    }


    float4* result_ptr = kernel_stream_.Process(work_buffer_, temp_buffer_, static_cast<int>(width),
                                                static_cast<int>(height),
                                                static_cast<size_t>(width), params_, stream);
    // Process() will synchronize the stream internally
    {
      const auto destroy_err = cudaStreamDestroy(stream);
      if (destroy_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamDestroy failed: ") +
                                 cudaGetErrorString(destroy_err));
      }
    }
          
    // float4* mapped_ptr = frame_sink_->MapResourceForWrite();
    // if (mapped_ptr) {
    //   size_t size_bytes = width * height * sizeof(float4);
    //   cudaMemcpy(mapped_ptr, result_ptr, size_bytes, cudaMemcpyDeviceToDevice);
    //   frame_sink_->UnmapResource();
    //   frame_sink_->NotifyFrameReady();
    // }

    if (output_img_) {
      output_img_->InitGPUData(width, height, CV_32FC4);
      cv::cuda::GpuMat output_gpu_mat = output_img_->GetGPUData();
      {
        const auto out_copy_err =
            cudaMemcpy2D(output_gpu_mat.ptr<float4>(), output_gpu_mat.step, result_ptr,
                         width * sizeof(float4), width * sizeof(float4), height,
                         cudaMemcpyDeviceToDevice);
        if (out_copy_err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy2D (work->output) failed: ") +
                                   cudaGetErrorString(out_copy_err));
        }
      }
      output_img_->SetGPUDataValid(true);
    }
  }
};
}  // namespace CUDA
};  // namespace puerhlab