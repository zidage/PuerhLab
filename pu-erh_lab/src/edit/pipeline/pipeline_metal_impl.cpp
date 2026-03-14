//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL

#include "edit/pipeline/pipeline_gpu_wrapper.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "edit/operators/GPU_kernels/fused_param.hpp"
#include "edit/operators/GPU_kernels/metal_param.hpp"
#include "edit/scope/scope_analyzer.hpp"
#include "image/image_buffer.hpp"
#include "image/metal_image.hpp"
#include "metal/compute_pipeline_cache.hpp"
#include "metal/metal_context.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {
namespace {
constexpr const char* kFusedPipelineKernelName          = "metal_fused_pipeline_rgba32f";
constexpr const char* kNeighborBlurHorizontalKernelName = "metal_neighbor_blur_h_rgba32f";
constexpr const char* kNeighborApplyVerticalKernelName  = "metal_neighbor_apply_v_rgba32f";
constexpr const char* kFusedPipelineDebugLabel          = "Metal fused pipeline";
constexpr const char* kNeighborBlurDebugLabel           = "Metal neighbor blur horizontal";
constexpr const char* kNeighborApplyDebugLabel          = "Metal neighbor apply vertical";
constexpr uint32_t    kMetalNeighborMaxRadius           = 20;
constexpr uint32_t    kMetalNeighborMaxTapCount         = 24;
constexpr auto        kReportInterval                   = std::chrono::milliseconds{500};
constexpr double      kFpsEmaAlpha                      = 0.15;

enum class MetalNeighborOpKind : uint32_t {
  Sharpen = 1,
  Clarity = 2,
};

struct alignas(16) MetalNeighborStageParams {
  uint32_t                                 kind_      = 0;
  uint32_t                                 radius_    = 0;
  uint32_t                                 tap_count_ = 0;
  float                                    amount_    = 0.0f;
  float                                    threshold_ = 0.0f;
  float                                    reserved_[3] = {};
  std::array<float, kMetalNeighborMaxTapCount> weights_ = {};
};

static_assert(sizeof(MetalNeighborStageParams) == 128,
              "MetalNeighborStageParams must stay ABI-compatible with Metal shaders.");

struct MetalNeighborStage {
  MetalNeighborStageParams params_ = {};
};

struct MetalExecutionStats {
  double input_prepare_ms    = 0.0;
  double fused_encode_ms     = 0.0;
  double neighbor_encode_ms  = 0.0;
  double gpu_wait_ms         = 0.0;
  double host_download_ms    = 0.0;
  double host_copy_submit_ms = 0.0;
  double output_wrap_ms      = 0.0;
  double total_ms            = 0.0;
  size_t detail_stage_count  = 0;
};

class MetalPreviewReporter {
 private:
  std::chrono::steady_clock::time_point last_report_time_{};
  double                                ema_fps_            = 0.0;
  double                                last_frame_ms_      = 0.0;
  double                                last_input_ms_      = 0.0;
  double                                last_fused_ms_      = 0.0;
  double                                last_neighbor_ms_   = 0.0;
  double                                last_gpu_wait_ms_   = 0.0;
  double                                last_download_ms_   = 0.0;
  double                                last_submit_ms_     = 0.0;
  double                                last_output_ms_     = 0.0;
  size_t                                last_stage_count_   = 0;
  size_t                                total_frames_       = 0;

 public:
  void Report(const MetalExecutionStats& stats) {
    const auto now = std::chrono::steady_clock::now();

    last_frame_ms_    = stats.total_ms;
    last_input_ms_    = stats.input_prepare_ms;
    last_fused_ms_    = stats.fused_encode_ms;
    last_neighbor_ms_ = stats.neighbor_encode_ms;
    last_gpu_wait_ms_ = stats.gpu_wait_ms;
    last_download_ms_ = stats.host_download_ms;
    last_submit_ms_   = stats.host_copy_submit_ms;
    last_output_ms_   = stats.output_wrap_ms;
    last_stage_count_ = stats.detail_stage_count;

    const double inst_fps = (stats.total_ms > 0.0) ? (1000.0 / stats.total_ms) : 0.0;
    ema_fps_              = (ema_fps_ <= 0.0)
                                ? inst_fps
                                : (ema_fps_ * (1.0 - kFpsEmaAlpha) + inst_fps * kFpsEmaAlpha);
    ++total_frames_;

    if (last_report_time_.time_since_epoch().count() == 0) {
      last_report_time_ = now;
    }
    if ((now - last_report_time_) < kReportInterval) {
      return;
    }

    static std::mutex print_mutex;
    std::lock_guard<std::mutex> guard(print_mutex);

    std::cout << "\r\033[2KMetal preview: " << std::fixed << std::setprecision(1) << ema_fps_
              << " fps"
              << " | last " << std::setprecision(2) << last_frame_ms_ << " ms"
              << " | parts in:" << last_input_ms_
              << " fe:" << last_fused_ms_
              << " ne:" << last_neighbor_ms_
              << " gw:" << last_gpu_wait_ms_
              << " hd:" << last_download_ms_
              << " hs:" << last_submit_ms_
              << " ow:" << last_output_ms_
              << " | stages " << last_stage_count_
              << " | frames " << total_frames_ << std::flush;

    last_report_time_ = now;
  }
};

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

auto MakeSharedBuffer(size_t length) -> NS::SharedPtr<MTL::Buffer> {
  auto* device = MetalContext::Instance().Device();
  if (device == nullptr) {
    throw std::runtime_error("Metal fused pipeline: Metal device is unavailable.");
  }

  auto buffer = NS::TransferPtr(
      device->newBuffer(static_cast<NS::UInteger>(length), MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("Metal fused pipeline: failed to allocate shared buffer.");
  }
  return buffer;
}

auto UploadStageParams(const MetalNeighborStageParams& params) -> NS::SharedPtr<MTL::Buffer> {
  auto buffer = MakeSharedBuffer(sizeof(MetalNeighborStageParams));
  std::memcpy(buffer->contents(), &params, sizeof(MetalNeighborStageParams));
  return buffer;
}

auto ResolveViewerDisplayConfig(const OperatorParams& params) -> ViewerDisplayConfig {
  return ViewerDisplayConfig{params.to_output_params_.encoding_space_,
                             params.to_output_params_.eotf_};
}

auto BuildGaussianWeights(float sigma, uint32_t radius)
    -> std::array<float, kMetalNeighborMaxTapCount> {
  std::array<float, kMetalNeighborMaxTapCount> weights{};
  const double safe_sigma  = std::max(static_cast<double>(sigma), 1.0e-4);
  const double inv2sigma2  = 0.5 / (safe_sigma * safe_sigma);
  double       full_weight = 1.0;

  weights[0] = 1.0f;
  for (uint32_t tap = 1; tap <= radius; ++tap) {
    const double w = std::exp(-(static_cast<double>(tap) * static_cast<double>(tap)) * inv2sigma2);
    weights[tap]   = static_cast<float>(w);
    full_weight += 2.0 * w;
  }

  if (full_weight > 0.0) {
    for (uint32_t tap = 0; tap <= radius; ++tap) {
      weights[tap] = static_cast<float>(static_cast<double>(weights[tap]) / full_weight);
    }
  }

  return weights;
}

auto BuildNeighborStageParams(MetalNeighborOpKind kind, float sigma, float amount, float threshold)
    -> MetalNeighborStageParams {
  MetalNeighborStageParams params;
  const float              safe_sigma = std::max(sigma, 1.0e-4f);

  params.kind_      = static_cast<uint32_t>(kind);
  params.radius_    = std::clamp<uint32_t>(static_cast<uint32_t>(std::ceil(3.0f * safe_sigma)), 1U,
                                           kMetalNeighborMaxRadius);
  params.tap_count_ = params.radius_ + 1U;
  params.amount_    = amount;
  params.threshold_ = threshold;
  params.weights_   = BuildGaussianWeights(safe_sigma, params.radius_);
  return params;
}

class MetalGPUPipeline final : public GPUPipelineImpl {
 private:
  std::shared_ptr<ImageBuffer> input_img_;
  OperatorParams*              cpu_params_ = nullptr;
  IFrameSink*                  frame_sink_ = nullptr;
  FusedOperatorParams          fused_params_ = {};
  metal::MetalFusedResources   resources_    = {};
  NS::SharedPtr<MTL::ComputePipelineState> fused_pipeline_                = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> neighbor_blur_horizontal_pipeline_ = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> neighbor_apply_vertical_pipeline_  = nullptr;
  MetalPreviewReporter         preview_reporter_;

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

  void EncodeNeighborBlurHorizontal(MTL::CommandBuffer* command_buffer, MTL::Buffer* stage_buffer,
                                    const metal::MetalImage& src, metal::MetalImage& dst) {
    if (!neighbor_blur_horizontal_pipeline_) {
      neighbor_blur_horizontal_pipeline_ =
          GetPipelineState(kNeighborBlurHorizontalKernelName, kNeighborBlurDebugLabel);
    }

    auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(neighbor_blur_horizontal_pipeline_.get());
    encoder->setTexture(src.Texture(), 0);
    encoder->setTexture(dst.Texture(), 1);
    encoder->setBuffer(stage_buffer, 0, 0);
    DispatchThreads(encoder.get(), neighbor_blur_horizontal_pipeline_.get(), src.Width(),
                    src.Height());
    encoder->endEncoding();
  }

  void EncodeNeighborApplyVertical(MTL::CommandBuffer* command_buffer, MTL::Buffer* stage_buffer,
                                   const metal::MetalImage& src,
                                   const metal::MetalImage& blur_horizontal,
                                   metal::MetalImage& dst) {
    if (!neighbor_apply_vertical_pipeline_) {
      neighbor_apply_vertical_pipeline_ =
          GetPipelineState(kNeighborApplyVerticalKernelName, kNeighborApplyDebugLabel);
    }

    auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(neighbor_apply_vertical_pipeline_.get());
    encoder->setTexture(src.Texture(), 0);
    encoder->setTexture(blur_horizontal.Texture(), 1);
    encoder->setTexture(dst.Texture(), 2);
    encoder->setBuffer(stage_buffer, 0, 0);
    DispatchThreads(encoder.get(), neighbor_apply_vertical_pipeline_.get(), src.Width(),
                    src.Height());
    encoder->endEncoding();
  }

  auto ShouldRunSharpen() const -> bool {
    return fused_params_.sharpen_enabled_ && fused_params_.sharpen_offset_ != 0.0f &&
           fused_params_.sharpen_radius_ > 0.0f;
  }

  auto ShouldRunClarity() const -> bool {
    return fused_params_.clarity_enabled_ && fused_params_.clarity_offset_ != 0.0f &&
           fused_params_.clarity_radius_ > 0.0f;
  }

  auto BuildNeighborStages() const -> std::vector<MetalNeighborStage> {
    std::vector<MetalNeighborStage> stages;
    stages.reserve(2);

    if (ShouldRunSharpen()) {
      stages.push_back(MetalNeighborStage{BuildNeighborStageParams(
          MetalNeighborOpKind::Sharpen, fused_params_.sharpen_radius_, fused_params_.sharpen_offset_,
          fused_params_.sharpen_threshold_)});
    }
    if (ShouldRunClarity()) {
      stages.push_back(MetalNeighborStage{BuildNeighborStageParams(
          MetalNeighborOpKind::Clarity, fused_params_.clarity_radius_, fused_params_.clarity_offset_,
          0.0f)});
    }

    return stages;
  }

  auto RunMetalPipeline(MetalExecutionStats& stats) -> metal::MetalImage {
    const auto input_prepare_start = std::chrono::steady_clock::now();
    EnsureMetalInput();
    const auto input_prepare_end = std::chrono::steady_clock::now();
    stats.input_prepare_ms =
        std::chrono::duration<double, std::milli>(input_prepare_end - input_prepare_start).count();

    const auto&               input           = input_img_->GetMetalImage();
    const auto                neighbor_stages = BuildNeighborStages();
    stats.detail_stage_count                  = neighbor_stages.size();

    metal::MetalImage working =
        metal::MetalImage::Create2D(input.Width(), input.Height(), input.Format(), true, true, false);
    metal::MetalImage blur_horizontal;
    metal::MetalImage scratch;
    if (!neighbor_stages.empty()) {
      blur_horizontal = metal::MetalImage::Create2D(input.Width(), input.Height(), input.Format(),
                                                    true, true, false);
      scratch = metal::MetalImage::Create2D(input.Width(), input.Height(), input.Format(), true,
                                            true, false);
    }

    auto command_buffer = MakeCommandBuffer();

    const auto fused_encode_start = std::chrono::steady_clock::now();
    EncodeFusedKernel(command_buffer.get(), input, working);
    const auto fused_encode_end = std::chrono::steady_clock::now();
    stats.fused_encode_ms =
        std::chrono::duration<double, std::milli>(fused_encode_end - fused_encode_start).count();

    metal::MetalImage* detail_src = &working;
    metal::MetalImage* detail_dst = &scratch;
    std::vector<NS::SharedPtr<MTL::Buffer>> stage_buffers;
    stage_buffers.reserve(neighbor_stages.size());

    for (const auto& stage : neighbor_stages) {
      stage_buffers.push_back(UploadStageParams(stage.params_));
      auto* stage_buffer = stage_buffers.back().get();

      const auto neighbor_encode_start = std::chrono::steady_clock::now();
      EncodeNeighborBlurHorizontal(command_buffer.get(), stage_buffer, *detail_src, blur_horizontal);
      EncodeNeighborApplyVertical(command_buffer.get(), stage_buffer, *detail_src, blur_horizontal,
                                  *detail_dst);
      const auto neighbor_encode_end = std::chrono::steady_clock::now();
      stats.neighbor_encode_ms +=
          std::chrono::duration<double, std::milli>(neighbor_encode_end - neighbor_encode_start)
              .count();

      std::swap(detail_src, detail_dst);
    }

    const auto gpu_wait_start = std::chrono::steady_clock::now();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    const auto gpu_wait_end = std::chrono::steady_clock::now();
    stats.gpu_wait_ms =
        std::chrono::duration<double, std::milli>(gpu_wait_end - gpu_wait_start).count();

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

    const auto         exec_start = std::chrono::steady_clock::now();
    MetalExecutionStats stats;
    metal::MetalImage   result = RunMetalPipeline(stats);
    const ViewerDisplayConfig display_config = ResolveViewerDisplayConfig(*cpu_params_);

    if (frame_sink_) {
      const auto submit_start = std::chrono::steady_clock::now();
#ifdef HAVE_METAL
      auto retained_texture =
          std::make_shared<NS::SharedPtr<MTL::Texture>>(NS::RetainPtr(result.Texture()));
      frame_sink_->SubmitFinalDisplayFrame(FinalDisplayFrameView{
          SharedGpuImageHandle{GpuBackend::Metal,
                               std::shared_ptr<void>(retained_texture, retained_texture.get()),
                               static_cast<int>(result.Width()),
                               static_cast<int>(result.Height()),
                               0,
                               FramePixelFormat::RGBA32F},
          static_cast<int>(result.Width()),
          static_cast<int>(result.Height()),
          FramePixelFormat::RGBA32F,
          display_config,
          AnalysisDomain::DisplayEncoded,
          {},
          0});
      frame_sink_->SubmitMetalFrame(
          ViewerMetalFrame{static_cast<int>(result.Width()), static_cast<int>(result.Height()),
                           reinterpret_cast<std::uintptr_t>(retained_texture->get()),
                           std::shared_ptr<const void>(retained_texture, retained_texture->get()),
                           display_config,
                           FramePresentationMode::FullFrame});
#else
      cv::Mat host_image;
      result.Download(host_image);
      stats.host_download_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                    submit_start)
              .count();
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
                      std::shared_ptr<const void>(host_pixels, host_pixels->data()), display_config,
                      FramePresentationMode::FullFrame});
#endif
      const auto submit_end = std::chrono::steady_clock::now();
      stats.host_copy_submit_ms =
          std::chrono::duration<double, std::milli>(submit_end - submit_start).count();
    }

    if (output_img) {
      const auto output_wrap_start = std::chrono::steady_clock::now();
      *output_img = ImageBuffer(std::move(result));
      const auto output_wrap_end = std::chrono::steady_clock::now();
      stats.output_wrap_ms =
          std::chrono::duration<double, std::milli>(output_wrap_end - output_wrap_start).count();
    }

    const auto exec_end = std::chrono::steady_clock::now();
    stats.total_ms      =
        std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
    preview_reporter_.Report(stats);
  }

  void ReleaseResources() override {
    resources_.Reset();
    fused_pipeline_                    = nullptr;
    neighbor_blur_horizontal_pipeline_ = nullptr;
    neighbor_apply_vertical_pipeline_  = nullptr;
  }
};

}  // namespace

auto CreateMetalGPUPipeline() -> std::unique_ptr<GPUPipelineImpl> {
  return std::make_unique<MetalGPUPipeline>();
}

}  // namespace puerhlab

#endif
