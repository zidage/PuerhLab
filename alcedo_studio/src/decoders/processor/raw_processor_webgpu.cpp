//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/raw_processor.hpp"

#ifdef HAVE_WEBGPU

#include <stdexcept>
#include <string>

#include "image/gpu_backend.hpp"
#include "webgpu/webgpu_context.hpp"

namespace alcedo {
namespace {

void EnsureWebGpuRawBackendAvailable() {
  auto& context = webgpu::WebGpuContext::Instance();
  if (context.IsAvailable()) {
    return;
  }

  std::string message = "RawProcessor: WebGPU backend is unavailable.";
  if (!context.InitializationLog().empty()) {
    message += " Dawn initialization log: " + context.InitializationLog();
  }
  throw std::runtime_error(message);
}

auto IsUnorientedOrIdentityFlip(const int flip) -> bool { return flip == 0 || flip == 1; }

}  // namespace

auto RawProcessor::ProcessDirectRgbWebGpu() -> ImageBuffer {
  EnsureWebGpuRawBackendAvailable();
  process_buffer_.SyncToGPU(GpuBackendKind::WebGPU);
  process_buffer_.ReleaseCPUData();

  if (!IsUnorientedOrIdentityFlip(raw_data_.sizes.flip)) {
    throw std::runtime_error(
        "RawProcessor: WebGPU direct RGB geometric corrections are not implemented yet.");
  }

  return {std::move(process_buffer_)};
}

auto RawProcessor::ProcessWebGpu() -> ImageBuffer {
  EnsureWebGpuRawBackendAvailable();
  if (input_kind_ == RawInputKind::DebayeredRgb) {
    return ProcessDirectRgbWebGpu();
  }

  throw std::runtime_error(
      "RawProcessor: WebGPU raw pipeline is routed, but raw linearization, debayer, highlight "
      "reconstruction, geometry, and RGBA pack operators are not implemented yet.");
}

}  // namespace alcedo

#endif
