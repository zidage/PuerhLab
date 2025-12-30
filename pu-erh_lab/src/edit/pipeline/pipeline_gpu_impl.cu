#include "edit/operators/GPU_kernels/basic.cuh"
#include "edit/operators/GPU_kernels/color.cuh"
#include "edit/operators/GPU_kernels/cst.cuh"
#include "edit/operators/GPU_kernels/detail.cuh"
#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/pipeline/gpu_scheduler.cuh"
#include "edit/pipeline/kernel_stream_gpu.cuh"
#include "edit/pipeline/pipeline_gpu_wrapper.hpp"

namespace puerhlab {
using namespace CUDA;

class GPUPipelineImpl {
 private:

  static constexpr auto                     BuildKernelStream = []() {
    auto to_ws  = GPU_TOWS_Kernel();
    auto exp    = GPU_ExposureOpKernel();
    auto cont   = GPU_ContrastOpKernel();
    auto white  = GPU_WhiteOpKernel();
    auto black  = GPU_BlackOpKernel();
    auto high   = GPU_HighlightOpKernel();
    auto shad   = GPU_ShadowOpKernel();

    auto tint   = GPU_TintOpKernel();
    auto sat    = GPU_SaturationOpKernel();
    auto vib    = GPU_VibranceOpKernel();
    auto hls    = GPU_HLSOpKernel();
    auto lmt    = GPU_LMT_Kernel();
    auto to_out = GPU_OUTPUT_Kernel();

    auto sharp  = GPU_SharpenKernel();
    auto clar   = GPU_ClarityKernel();

    return GPU_StaticKernelStream(GPU_PointChain(to_ws, exp, cont, white, black, high, shad, tint,
                                                                     sat, vib, hls, lmt, to_out),
                                                    sharp, clar);
  };

  using StaticKernelStreamType = decltype(BuildKernelStream());
  StaticKernelStreamType _static_kernel_stream = BuildKernelStream();
  GPU_KernelLauncher<StaticKernelStreamType> _launcher;

 public:
  GPUPipelineImpl() : _launcher(nullptr, _static_kernel_stream) {}

  void SetInput(std::shared_ptr<ImageBuffer> input_img) { _launcher.SetInputImage(input_img); }

  void SetParams(OperatorParams& cpu_params) {
    _launcher.SetParams(cpu_params);
  }

  void Execute(std::shared_ptr<ImageBuffer> output_img) {
    _launcher.SetOutputImage(output_img);
    _launcher.Execute();
  }
  
};

GPUPipelineWrapper::GPUPipelineWrapper() : _impl(std::make_unique<GPUPipelineImpl>()) {}

GPUPipelineWrapper::~GPUPipelineWrapper() = default;

void GPUPipelineWrapper::SetInputImage(std::shared_ptr<ImageBuffer> input_image) {
  _impl->SetInput(input_image);
}

void GPUPipelineWrapper::SetParams(OperatorParams& cpu_params) {
  _impl->SetParams(cpu_params);
}

void GPUPipelineWrapper::Execute(std::shared_ptr<ImageBuffer> output) { _impl->Execute(output); }
};  // namespace puerhlab