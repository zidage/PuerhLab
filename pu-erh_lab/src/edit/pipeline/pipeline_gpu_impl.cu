//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "edit/operators/GPU_kernels/basic.cuh"
#include "edit/operators/GPU_kernels/color.cuh"
#include "edit/operators/GPU_kernels/cst.cuh"
#include "edit/operators/GPU_kernels/detail.cuh"
#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/pipeline/gpu_scheduler.cuh"
#include "edit/pipeline/kernel_stream_gpu.cuh"
#include "edit/pipeline/pipeline_gpu_wrapper.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {
using namespace CUDA;

class GPUPipelineImpl {
 private:
  static constexpr auto BuildKernelStream = []() {
    auto to_ws  = GPU_TOWS_Kernel();
    auto exp    = GPU_ExposureOpKernel();
    auto cont   = GPU_ContrastOpKernel();
    auto tone  = GPU_ToneOpKernel();
    auto high   = GPU_HighlightOpKernel();
    auto shad   = GPU_ShadowOpKernel();
    auto curve  = GPU_CurveOpKernel();

    auto sat    = GPU_SaturationOpKernel();
    auto vib    = GPU_VibranceOpKernel();
    auto wheel  = GPU_ColorWheelOpKernel();
    auto hls    = GPU_HLSOpKernel();
    auto lmt    = GPU_LMT_Kernel();
    auto to_out = GPU_OUTPUT_Kernel();

    auto sharp  = GPU_SharpenKernel();
    auto clar   = GPU_ClarityKernel();

    return GPU_StaticKernelStream(GPU_PointChain(to_ws, exp, cont, tone, high, shad, curve, sat,
                                                 vib, wheel, hls, lmt, to_out),
                                  sharp, clar);
  };

  using StaticKernelStreamType                                     = decltype(BuildKernelStream());
  StaticKernelStreamType                     static_kernel_stream_ = BuildKernelStream();
  GPU_KernelLauncher<StaticKernelStreamType> launcher_;

 public:
  GPUPipelineImpl() : launcher_(nullptr, static_kernel_stream_) {}

  void SetInput(std::shared_ptr<ImageBuffer> input_img) { launcher_.SetInputImage(input_img); }

  void SetParams(OperatorParams& cpu_params) { launcher_.SetParams(cpu_params); }

  void SetFrameSink(IFrameSink* frame_sink) { launcher_.SetFrameSink(frame_sink); }

  void Execute(std::shared_ptr<ImageBuffer> output_img) {
    launcher_.SetOutputImage(output_img);
    launcher_.Execute();
  }

  void ReleaseResources() { launcher_.ReleaseResources(); }
};

GPUPipelineWrapper::GPUPipelineWrapper() : impl_(std::make_unique<GPUPipelineImpl>()) {}

GPUPipelineWrapper::~GPUPipelineWrapper() {impl_->ReleaseResources();};

void GPUPipelineWrapper::SetInputImage(std::shared_ptr<ImageBuffer> input_image) {
  impl_->SetInput(input_image);
}

void GPUPipelineWrapper::SetParams(OperatorParams& cpu_params) { impl_->SetParams(cpu_params); }

void GPUPipelineWrapper::SetFrameSink(IFrameSink* frame_sink) { impl_->SetFrameSink(frame_sink); }

void GPUPipelineWrapper::Execute(std::shared_ptr<ImageBuffer> output) { impl_->Execute(output); }

void GPUPipelineWrapper::ReleaseResources() { impl_->ReleaseResources(); }
};  // namespace puerhlab
