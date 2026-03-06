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

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>
#include <string>

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
  std::shared_ptr<ImageBuffer>               input_img_;

  float4*                                    work_buffer_    = nullptr;
  float4*                                    temp_buffer_    = nullptr;
  size_t                                     allocated_size_ = 0;

  std::shared_ptr<ImageBuffer>               output_img_;

  KernelStreamT                              kernel_stream_;

  cudaStream_t                               stream_ = nullptr;

  GPUOperatorParams                          params_;

  IFrameSink*                                frame_sink_ = nullptr;

  // Lightweight FPS reporter (per launcher instance). Prints a single updating
  // line in the CLI, throttled to avoid spamming.
  std::chrono::steady_clock::time_point      last_report_time_{};
  double                                     ema_fps_               = 0.0;
  double                                     last_frame_ms_         = 0.0;
  double                                     last_ensure_size_ms_   = 0.0;
  double                                     last_input_copy_enqueue_ms_ = 0.0;
  double                                     last_kernel_dispatch_ms_ = 0.0;
  double                                     last_present_copy_enqueue_ms_ = 0.0;
  double                                     last_present_sync_ms_  = 0.0;
  double                                     last_output_copy_enqueue_ms_ = 0.0;
  double                                     last_output_sync_ms_   = 0.0;
  size_t                                     frames_since_report_   = 0;
  size_t                                     total_frames_rendered_ = 0;

  static constexpr std::chrono::milliseconds kReportInterval{500};
  static constexpr double                    kEmaAlpha = 0.15;  // smoothing factor

 public:
  GPU_KernelLauncher(std::shared_ptr<ImageBuffer> input_img, KernelStreamT kernel_stream)
      : input_img_(input_img), kernel_stream_(kernel_stream) {
    const auto stream_err = cudaStreamCreate(&stream_);
    if (stream_err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                               cudaGetErrorString(stream_err));
    }
  }

  void ReleaseScratchBuffers() {
    if (stream_) {
      (void)cudaStreamSynchronize(stream_);
    }
    if (work_buffer_) {
      cudaFree(work_buffer_);
      work_buffer_ = nullptr;
    }
    if (temp_buffer_) {
      cudaFree(temp_buffer_);
      temp_buffer_ = nullptr;
    }
    allocated_size_ = 0;
  }

  void ReleaseResources() {
    ReleaseScratchBuffers();
    params_.to_ws_lut_.Reset();
    params_.lmt_lut_.Reset();
    params_.to_output_lut_.Reset();
    params_.to_output_params_.odt_params_.Reset();
  }

  ~GPU_KernelLauncher() {
    ReleaseResources();
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  void SetInputImage(std::shared_ptr<ImageBuffer> input_img) {
    input_img_ = input_img;
    if (!input_img_) {
      throw std::runtime_error("GPU_KernelLauncher: input image is null.");
    }
    if (input_img_ && !input_img_->gpu_data_valid_ && input_img_->cpu_data_valid_) {
      input_img_->SyncToGPU();
    }
    cv::cuda::GpuMat gpu_mat = input_img_->GetGPUData();
    if (gpu_mat.type() != CV_32FC4) {
      throw std::runtime_error(
          std::string("GPU_KernelLauncher: expected input type CV_32FC4, got type ") +
          std::to_string(gpu_mat.type()));
    }
    

    size_t           width       = gpu_mat.cols;
    size_t           height      = gpu_mat.rows;
    size_t           needed_size = width * height * sizeof(float4);

    if (needed_size > allocated_size_) {
      if (work_buffer_) {
        cudaFree(work_buffer_);
        work_buffer_ = nullptr;
      }
      if (temp_buffer_) {
        cudaFree(temp_buffer_);
        temp_buffer_ = nullptr;
      }

      const auto work_alloc_err = cudaMalloc((void**)&work_buffer_, needed_size);
      if (work_alloc_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc (work_buffer_) failed: ") +
                                 cudaGetErrorString(work_alloc_err));
      }
      const auto temp_alloc_err = cudaMalloc((void**)&temp_buffer_, needed_size);
      if (temp_alloc_err != cudaSuccess) {
        cudaFree(work_buffer_);
        work_buffer_ = nullptr;
        throw std::runtime_error(std::string("cudaMalloc (temp_buffer_) failed: ") +
                                 cudaGetErrorString(temp_alloc_err));
      }
      allocated_size_ = needed_size;
    }
  }

  void SetOutputImage(std::shared_ptr<ImageBuffer> output_img) { output_img_ = output_img; }

  void SetParams(OperatorParams& cpu_params) {
    params_ = GPUParamsConverter::ConvertFromCPU(cpu_params, params_);
  }

  void SetFrameSink(IFrameSink* frame_sink) { frame_sink_ = frame_sink; }

  void Execute() {
    const auto exec_start = std::chrono::steady_clock::now();
    double     ensure_size_ms = 0.0;
    double     input_copy_enqueue_ms = 0.0;
    double     kernel_dispatch_ms = 0.0;
    double     present_copy_enqueue_ms = 0.0;
    double     present_sync_ms = 0.0;
    double     output_copy_enqueue_ms = 0.0;
    double     output_sync_ms = 0.0;
    if (!input_img_ || !work_buffer_) {
      throw std::runtime_error("Input image not set or work buffer not allocated.");
    }

    if (!stream_) {
      throw std::runtime_error("CUDA stream not initialized.");
    }

    cv::cuda::GpuMat gpu_mat = input_img_->GetGPUData();
    if (gpu_mat.type() != CV_32FC4) {
      throw std::runtime_error(
          std::string("GPU_KernelLauncher: expected execution input type CV_32FC4, got type ") +
          std::to_string(gpu_mat.type()));
    }
    size_t           width   = gpu_mat.cols;
    size_t           height  = gpu_mat.rows;

    if (frame_sink_) {
      const auto ensure_size_start = std::chrono::steady_clock::now();
      frame_sink_->EnsureSize(width, height);
      const auto ensure_size_end   = std::chrono::steady_clock::now();
      ensure_size_ms               =
          std::chrono::duration<double, std::milli>(ensure_size_end - ensure_size_start).count();
    }

    {
      const auto copy_start = std::chrono::steady_clock::now();
      const auto copy_err = cudaMemcpy2DAsync(
          work_buffer_, width * sizeof(float4), gpu_mat.ptr<float4>(), gpu_mat.step,
          width * sizeof(float4), height, cudaMemcpyDeviceToDevice, stream_);
      const auto copy_end = std::chrono::steady_clock::now();
      input_copy_enqueue_ms =
          std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
      if (copy_err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(copy_err) << std::endl;
        throw std::runtime_error(std::string("cudaMemcpy2D (input->work) failed: ") +
                                 cudaGetErrorString(copy_err));
      }
    }

    const auto kernel_dispatch_start = std::chrono::steady_clock::now();
    float4* result_ptr = kernel_stream_.Process(work_buffer_, temp_buffer_, static_cast<int>(width),
                                                static_cast<int>(height),
                                                static_cast<size_t>(width), params_, stream_,
                                                /*sync=*/false);
    const auto kernel_dispatch_end = std::chrono::steady_clock::now();
    kernel_dispatch_ms = std::chrono::duration<double, std::milli>(kernel_dispatch_end -
                                                                    kernel_dispatch_start)
                             .count();
    // Synchronize once later, right before presenting.

    if (frame_sink_) {
      float4* mapped_ptr = frame_sink_->MapResourceForWrite();
      if (mapped_ptr) {
        size_t     size_bytes = width * height * sizeof(float4);
        const auto present_copy_start = std::chrono::steady_clock::now();
        const auto out_copy_err =
            cudaMemcpyAsync(mapped_ptr, result_ptr, size_bytes, cudaMemcpyDeviceToDevice, stream_);
        const auto present_copy_end = std::chrono::steady_clock::now();
        present_copy_enqueue_ms =
            std::chrono::duration<double, std::milli>(present_copy_end - present_copy_start).count();
        if (out_copy_err != cudaSuccess) {
          frame_sink_->UnmapResource();
          throw std::runtime_error(std::string("cudaMemcpyAsync (work->frame) failed: ") +
                                   cudaGetErrorString(out_copy_err));
        }

        const auto present_sync_start = std::chrono::steady_clock::now();
        const auto sync_err = cudaStreamSynchronize(stream_);
        const auto present_sync_end = std::chrono::steady_clock::now();
        present_sync_ms =
            std::chrono::duration<double, std::milli>(present_sync_end - present_sync_start).count();
        if (sync_err != cudaSuccess) {
          frame_sink_->UnmapResource();
          throw std::runtime_error(std::string("cudaStreamSynchronize (present) failed: ") +
                                   cudaGetErrorString(sync_err));
        }

        frame_sink_->UnmapResource();
        frame_sink_->NotifyFrameReady();

        // FPS/frametime reporting
        const auto   exec_end = std::chrono::steady_clock::now();
        const double frame_ms =
            std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
        last_frame_ms_        = frame_ms;
        last_ensure_size_ms_  = ensure_size_ms;
        last_input_copy_enqueue_ms_ = input_copy_enqueue_ms;
        last_kernel_dispatch_ms_ = kernel_dispatch_ms;
        last_present_copy_enqueue_ms_ = present_copy_enqueue_ms;
        last_present_sync_ms_ = present_sync_ms;
        last_output_copy_enqueue_ms_ = output_copy_enqueue_ms;
        last_output_sync_ms_ = output_sync_ms;
        const double inst_fps = (frame_ms > 0.0) ? (1000.0 / frame_ms) : 0.0;
        ema_fps_ =
            (ema_fps_ <= 0.0) ? inst_fps : (ema_fps_ * (1.0 - kEmaAlpha) + inst_fps * kEmaAlpha);
        ++frames_since_report_;
        ++total_frames_rendered_;

        if (last_report_time_.time_since_epoch().count() == 0) {
          last_report_time_ = exec_end;
        }

        if ((exec_end - last_report_time_) >= kReportInterval) {
          static std::mutex           print_mutex;
          std::lock_guard<std::mutex> guard(print_mutex);

          std::cout << "\r\033[2KGPU preview: " << std::fixed << std::setprecision(1) << ema_fps_
                    << " fps"
                    << " | last " << std::setprecision(2) << last_frame_ms_ << " ms"
                    << " | parts e:" << std::setprecision(2) << last_ensure_size_ms_
                    << " in:" << last_input_copy_enqueue_ms_
                    << " k:" << last_kernel_dispatch_ms_
                    << " pc:" << last_present_copy_enqueue_ms_
                    << " ps:" << last_present_sync_ms_
                    << " oc:" << last_output_copy_enqueue_ms_
                    << " os:" << last_output_sync_ms_
                    << " | frames " << total_frames_rendered_ << std::flush;

          frames_since_report_ = 0;
          last_report_time_    = exec_end;
        }
      }
    }

    if (output_img_) {
      output_img_->InitGPUData(width, height, CV_32FC4);
      cv::cuda::GpuMat output_gpu_mat = output_img_->GetGPUData();
      {
        const auto output_copy_start = std::chrono::steady_clock::now();
        const auto out_copy_err = cudaMemcpy2DAsync(
            output_gpu_mat.ptr<float4>(), output_gpu_mat.step, result_ptr, width * sizeof(float4),
            width * sizeof(float4), height, cudaMemcpyDeviceToDevice, stream_);
        const auto output_copy_end = std::chrono::steady_clock::now();
        output_copy_enqueue_ms = std::chrono::duration<double, std::milli>(output_copy_end -
                                                                            output_copy_start)
                                     .count();
        if (out_copy_err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy2DAsync (work->output) failed: ") +
                                   cudaGetErrorString(out_copy_err));
        }
      }
      {
        const auto output_sync_start = std::chrono::steady_clock::now();
        const auto sync_err = cudaStreamSynchronize(stream_);
        const auto output_sync_end = std::chrono::steady_clock::now();
        output_sync_ms = std::chrono::duration<double, std::milli>(output_sync_end -
                                                                    output_sync_start)
                             .count();
        if (sync_err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaStreamSynchronize (output copy) failed: ") +
                                   cudaGetErrorString(sync_err));
        }
      }
      output_img_->SetGPUDataValid(true);
    }
  }
};
}  // namespace CUDA
};  // namespace puerhlab
