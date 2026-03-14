//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "edit/scope/scope_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>

#include "edit/scope/detail/scope_cuda_shared.cuh"
#endif

namespace puerhlab {
namespace {

class NullScopeAnalyzer final : public IScopeAnalyzer {
 public:
  void SubmitFrame(const FinalDisplayFrameView&, const ScopeRequest&) override {}

  auto GetLatestOutput() -> ScopeOutputSet override { return {}; }

  void ResizeResources(const ScopeRequest&) override {}

  void ReleaseResources() override {}
};

auto NormalizeHistogramToUnitRange(const std::vector<uint32_t>& counts, int bins)
    -> ScopeHistogramRenderData {
  ScopeHistogramRenderData data;
  if (bins <= 0 || counts.size() < static_cast<size_t>(bins * 3)) {
    return data;
  }

  const auto max_it = std::max_element(counts.begin(), counts.begin() + static_cast<size_t>(bins * 3));
  const float denom = (max_it != counts.end() && *max_it > 0U) ? static_cast<float>(*max_it) : 1.0f;

  data.bins = bins;
  data.rgb.resize(static_cast<size_t>(bins * 3), 0.0f);
  for (size_t i = 0; i < data.rgb.size(); ++i) {
    data.rgb[i] = static_cast<float>(counts[i]) / denom;
  }
  data.valid = true;
  return data;
}

auto NormalizeWaveformToUnitRange(const std::vector<float>& rgba, int width, int height)
    -> ScopeWaveformRenderData {
  ScopeWaveformRenderData data;
  if (width <= 0 || height <= 0 ||
      rgba.size() < static_cast<size_t>(width) * static_cast<size_t>(height) * 4U) {
    return data;
  }

  float max_value = 0.0f;
  for (float value : rgba) {
    max_value = std::max(max_value, value);
  }
  if (max_value <= std::numeric_limits<float>::epsilon()) {
    max_value = 1.0f;
  }

  data.width  = width;
  data.height = height;
  data.rgba.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4U, 0.0f);
  for (size_t i = 0; i < data.rgba.size(); ++i) {
    data.rgba[i] = std::clamp(rgba[i] / max_value, 0.0f, 1.0f);
  }
  data.valid = true;
  return data;
}

}  // namespace

#ifdef HAVE_CUDA
auto CreateCudaScopeAnalyzer() -> std::shared_ptr<IScopeAnalyzer>;
#endif

auto CreateMetalScopeAnalyzer() -> std::shared_ptr<IScopeAnalyzer> {
  return std::make_shared<NullScopeAnalyzer>();
}

auto CreateDefaultScopeAnalyzer() -> std::shared_ptr<IScopeAnalyzer> {
#ifdef HAVE_CUDA
  return CreateCudaScopeAnalyzer();
#elif defined(HAVE_METAL)
  return CreateMetalScopeAnalyzer();
#else
  return std::make_shared<NullScopeAnalyzer>();
#endif
}

auto ReadScopeRenderSnapshot(const ScopeOutputSet& output) -> ScopeRenderSnapshot {
  ScopeRenderSnapshot snapshot;
  snapshot.generation = output.generation;

#ifdef HAVE_CUDA
  if (output.histogram_valid && output.histogram_buffer &&
      output.histogram_buffer.backend == GpuBackend::Cuda) {
    const auto* resource = static_cast<const scope::cuda_detail::CudaDeviceBufferResource*>(
        output.histogram_buffer.resource.get());
    if (resource && resource->device_ptr && output.histogram_bins > 0) {
      std::vector<uint32_t> counts(static_cast<size_t>(output.histogram_bins) * 3U, 0U);
      const size_t expected_bytes = counts.size() * sizeof(uint32_t);
      if (resource->size_bytes >= expected_bytes &&
          cudaMemcpy(counts.data(), resource->device_ptr, expected_bytes, cudaMemcpyDeviceToHost) ==
              cudaSuccess) {
        snapshot.histogram = NormalizeHistogramToUnitRange(counts, output.histogram_bins);
      }
    }
  }

  if (output.waveform_valid && output.waveform_image &&
      output.waveform_image.backend == GpuBackend::Cuda) {
    const auto* resource = static_cast<const scope::cuda_detail::CudaLinearImageResource*>(
        output.waveform_image.resource.get());
    if (resource && resource->device_ptr && output.waveform_width > 0 && output.waveform_height > 0) {
      std::vector<float> rgba(static_cast<size_t>(output.waveform_width) *
                                  static_cast<size_t>(output.waveform_height) * 4U,
                              0.0f);
      const size_t row_bytes = static_cast<size_t>(output.waveform_width) * sizeof(float) * 4U;
      const auto copy_status =
          cudaMemcpy2D(rgba.data(), row_bytes, resource->device_ptr, resource->row_bytes, row_bytes,
                       static_cast<size_t>(output.waveform_height), cudaMemcpyDeviceToHost);
      if (copy_status == cudaSuccess) {
        snapshot.waveform =
            NormalizeWaveformToUnitRange(rgba, output.waveform_width, output.waveform_height);
      }
    }
  }
#else
  (void)output;
#endif

  return snapshot;
}

}  // namespace puerhlab
