//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "edit/pipeline/pipeline_gpu_wrapper.hpp"

#include <stdexcept>
#include <utility>

namespace puerhlab {
#ifdef HAVE_CUDA
auto CreateCUDAGPUPipeline() -> std::unique_ptr<GPUPipelineImpl>;
#endif
#ifdef HAVE_METAL
auto CreateMetalGPUPipeline() -> std::unique_ptr<GPUPipelineImpl>;
#endif

namespace {
class UnavailableGPUPipeline final : public GPUPipelineImpl {
 public:
  void SetInputImage(std::shared_ptr<ImageBuffer>) override {}

  void SetParams(OperatorParams&) override {}

  void SetFrameSink(IFrameSink*) override {}

  void Execute(std::shared_ptr<ImageBuffer>) override {
    throw std::runtime_error("GPU backend unavailable: compiled GPU pipeline implementation is missing.");
  }

  void ReleaseResources() override {}
};

auto CreateDefaultGPUPipeline() -> std::unique_ptr<GPUPipelineImpl> {
#ifdef HAVE_CUDA
  return CreateCUDAGPUPipeline();
#elif defined(HAVE_METAL)
  return CreateMetalGPUPipeline();
#else
  return std::make_unique<UnavailableGPUPipeline>();
#endif
}
}  // namespace

GPUPipelineWrapper::GPUPipelineWrapper() : impl_(CreateDefaultGPUPipeline()) {}

GPUPipelineWrapper::~GPUPipelineWrapper() {
  if (impl_) {
    impl_->ReleaseResources();
  }
}

void GPUPipelineWrapper::SetInputImage(std::shared_ptr<ImageBuffer> input_image) {
  impl_->SetInputImage(std::move(input_image));
}

void GPUPipelineWrapper::SetParams(OperatorParams& params) { impl_->SetParams(params); }

void GPUPipelineWrapper::SetFrameSink(IFrameSink* frame_sink) { impl_->SetFrameSink(frame_sink); }

void GPUPipelineWrapper::Execute(std::shared_ptr<ImageBuffer> output) {
  impl_->Execute(std::move(output));
}

auto GPUPipelineWrapper::HasAcceleratedBackend() const -> bool {
#if defined(HAVE_CUDA) || defined(HAVE_METAL)
  return true;
#else
  return false;
#endif
}

void GPUPipelineWrapper::ReleaseResources() { impl_->ReleaseResources(); }
}  // namespace puerhlab
