//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "edit/pipeline/pipeline_gpu_wrapper.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "edit/operators/GPU_kernels/fused_param.hpp"
#include "edit/operators/GPU_kernels/metal_param.hpp"
#include "image/image_buffer.hpp"
#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {
namespace {
constexpr const char* kFusedPipelineKernelName   = "metal_fused_pipeline_rgba32f";
constexpr const char* kSharpenKernelName         = "metal_detail_sharpen_rgba32f";
constexpr const char* kClarityKernelName         = "metal_detail_clarity_rgba32f";
constexpr const char* kFusedPipelineDebugLabel   = "Metal fused pipeline";
constexpr const char* kSharpenPipelineDebugLabel = "Metal detail sharpen";
constexpr const char* kClarityPipelineDebugLabel = "Metal detail clarity";

auto MakeCommandBuffer() -> NS::SharedPtr<MTL::CommandBuffer> {
  auto* queue = MetalContext::Instance().Queue();
  if (queue == nullptr) {
    throw std::runtime_error("Metal fused pipeline: Metal queue is unavailable.");
  }
  auto command_buffer = NS::RetainPtr(queue->commandBuffer());
  if (!command_buffer) {
    throw std::runtime_error("Metal fused pipeline: failed to create command buffer.");
  }
  return command_buffer;
}

void DispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline,
                     uint32_t width, uint32_t height) {
  const auto thread_width  = std::max<NS::UInteger>(1, pipeline->threadExecutionWidth());
  const auto thread_height =
      std::max<NS::UInteger>(1, pipeline->maxTotalThreadsPerThreadgroup() / thread_width);
  const MTL::Size threads_per_group{thread_width, thread_height, 1};
  const MTL::Size threads_per_grid{width, height, 1};
  encoder->dispatchThreads(threads_per_grid, threads_per_group);
}

class MetalGPUPipeline final : public GPUPipelineImpl {
 private:
  std::shared_ptr<ImageBuffer> input_img_;
  OperatorParams*              cpu_params_ = nullptr;
  IFrameSink*                  frame_sink_ = nullptr;
  FusedOperatorParams          fused_params_         = {};
  metal::MetalFusedResources   resources_            = {};
  NS::SharedPtr<MTL::ComputePipelineState> fused_pipeline_   = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> sharpen_pipeline_ = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> clarity_pipeline_ = nullptr;

  void EnsureMetalInput() {
    if (!input_img_) {
      throw std::runtime_error("Metal fused pipeline: input image is null.");
    }
    if (!input_img_->gpu_data_valid_) {
      if (!input_img_->cpu_data_valid_) {
        throw std::runtime_error("Metal fused pipeline: input image has no valid CPU or GPU data.");
      }
      input_img_->SyncToGPU();
    }
    if (input_img_->GetGPUType() != CV_32FC4) {
      input_img_->ConvertGPUDataTo(CV_32FC4);
    }
  }

  auto GetPipelineState(const char* kernel_name, const char* debug_label)
      -> NS::SharedPtr<MTL::ComputePipelineState> {
#ifndef PUERHLAB_METAL_FUSED_PIPELINE_METALLIB_PATH
    throw std::runtime_error("Metal fused pipeline metallib path is not configured.");
#else
    return metal::ComputePipelineCache::Instance().GetPipelineState(
        PUERHLAB_METAL_FUSED_PIPELINE_METALLIB_PATH, kernel_name, debug_label);
#endif
  }

  void EncodeFusedKernel(MTL::CommandBuffer* command_buffer, const metal::MetalImage& src,
                         metal::MetalImage& dst) {
    if (!fused_pipeline_) {
      fused_pipeline_ = GetPipelineState(kFusedPipelineKernelName, kFusedPipelineDebugLabel);
    }
    auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(fused_pipeline_.get());
    encoder->setTexture(src.Texture(), 0);
    encoder->setTexture(dst.Texture(), 1);
    encoder->setBuffer(resources_.params_buffer_.get(), 0, 0);
    encoder->setBuffer(resources_.lmt_lut_.buffer_.get(), 0, 1);
    DispatchThreads(encoder.get(), fused_pipeline_.get(), src.Width(), src.Height());
    encoder->endEncoding();
  }

  void EncodeDetailKernel(MTL::CommandBuffer* command_buffer, MTL::ComputePipelineState* pipeline,
                          const metal::MetalImage& src, metal::MetalImage& dst) {
    auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipeline);
    encoder->setTexture(src.Texture(), 0);
    encoder->setTexture(dst.Texture(), 1);
    encoder->setBuffer(resources_.params_buffer_.get(), 0, 0);
    DispatchThreads(encoder.get(), pipeline, src.Width(), src.Height());
    encoder->endEncoding();
  }

  auto GetSharpenPipeline() -> NS::SharedPtr<MTL::ComputePipelineState> {
    if (!sharpen_pipeline_) {
      sharpen_pipeline_ = GetPipelineState(kSharpenKernelName, kSharpenPipelineDebugLabel);
    }
    return sharpen_pipeline_;
  }

  auto GetClarityPipeline() -> NS::SharedPtr<MTL::ComputePipelineState> {
    if (!clarity_pipeline_) {
      clarity_pipeline_ = GetPipelineState(kClarityKernelName, kClarityPipelineDebugLabel);
    }
    return clarity_pipeline_;
  }

  auto ShouldRunSharpen() const -> bool {
    return fused_params_.sharpen_enabled_ && fused_params_.sharpen_offset_ != 0.0f &&
           fused_params_.sharpen_radius_ > 0.0f;
  }

  auto ShouldRunClarity() const -> bool {
    return fused_params_.clarity_enabled_ && fused_params_.clarity_radius_ > 0.0f;
  }

  auto RunMetalPipeline() -> metal::MetalImage {
    EnsureMetalInput();
    const auto& input   = input_img_->GetMetalImage();
    const bool  sharpen = ShouldRunSharpen();
    const bool  clarity = ShouldRunClarity();

    metal::MetalImage working =
        metal::MetalImage::Create2D(input.Width(), input.Height(), input.Format(), true, true, false);
    metal::MetalImage scratch;
    if (sharpen || clarity) {
      scratch = metal::MetalImage::Create2D(input.Width(), input.Height(), input.Format(), true,
                                            true, false);
    }

    auto command_buffer = MakeCommandBuffer();
    EncodeFusedKernel(command_buffer.get(), input, working);

    metal::MetalImage* detail_src = &working;
    metal::MetalImage* detail_dst = &scratch;

    if (sharpen) {
      auto pipeline = GetSharpenPipeline();
      EncodeDetailKernel(command_buffer.get(), pipeline.get(), *detail_src, *detail_dst);
      std::swap(detail_src, detail_dst);
    }
    if (clarity) {
      auto pipeline = GetClarityPipeline();
      EncodeDetailKernel(command_buffer.get(), pipeline.get(), *detail_src, *detail_dst);
      std::swap(detail_src, detail_dst);
    }

    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    return *detail_src;
  }

 public:
  void SetInputImage(std::shared_ptr<ImageBuffer> input_image) override {
    input_img_ = std::move(input_image);
  }

  void SetParams(OperatorParams& params) override {
    cpu_params_    = &params;
    fused_params_  = FusedParamsConverter::ConvertFromCPU(params, fused_params_);
    resources_     = metal::MetalFusedParamUploader::Upload(fused_params_, params, resources_);
  }

  void SetFrameSink(IFrameSink* frame_sink) override { frame_sink_ = frame_sink; }

  void Execute(std::shared_ptr<ImageBuffer> output_img) override {
    if (!cpu_params_) {
      throw std::runtime_error("Metal fused pipeline: parameters were not set.");
    }

    metal::MetalImage result = RunMetalPipeline();
    if (frame_sink_) {
      cv::Mat host_image;
      result.Download(host_image);
      if (host_image.type() != CV_32FC4) {
        throw std::runtime_error("Metal fused pipeline: expected RGBA32F host frame for viewer.");
      }

      const size_t row_bytes =
          static_cast<size_t>(host_image.cols) * static_cast<size_t>(sizeof(cv::Vec4f));
      auto host_pixels =
          std::make_shared<std::vector<float>>(static_cast<size_t>(host_image.cols) *
                                               static_cast<size_t>(host_image.rows) * 4U);
      cv::Mat contiguous_host(host_image.rows, host_image.cols, CV_32FC4, host_pixels->data(),
                              row_bytes);
      host_image.copyTo(contiguous_host);

      frame_sink_->SubmitHostFrame(
          ViewerFrame{host_image.cols, host_image.rows, row_bytes,
                      std::shared_ptr<const void>(host_pixels, host_pixels->data()),
                      FramePresentationMode::FullFrame});
    }

    if (output_img) {
      *output_img = ImageBuffer(std::move(result));
    }
  }

  void ReleaseResources() override {
    resources_.Reset();
    fused_pipeline_   = nullptr;
    sharpen_pipeline_ = nullptr;
    clarity_pipeline_ = nullptr;
  }
};

}  // namespace

auto CreateMetalGPUPipeline() -> std::unique_ptr<GPUPipelineImpl> {
  return std::make_unique<MetalGPUPipeline>();
}

}  // namespace puerhlab

#endif
