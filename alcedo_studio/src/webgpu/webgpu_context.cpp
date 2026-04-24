//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_WEBGPU

#include "webgpu/webgpu_context.hpp"

#include <dawn/native/DawnNative.h>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace alcedo {
namespace webgpu {
namespace {

struct AdapterAttempt {
  wgpu::BackendType  backend_type;
  wgpu::FeatureLevel feature_level;
  const char*        name;
};

auto StringViewToString(WGPUStringView view) -> std::string {
  if (view.data == nullptr) {
    return {};
  }
  if (view.length == WGPU_STRLEN) {
    return std::string(view.data);
  }
  return std::string(view.data, view.length);
}

auto AdapterInfoLabel(WGPUAdapter adapter) -> std::string {
  WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
  if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
    return "unknown adapter";
  }

  const auto         vendor      = StringViewToString(info.vendor);
  const auto         device      = StringViewToString(info.device);
  const auto         description = StringViewToString(info.description);

  std::ostringstream label;
  if (!description.empty()) {
    label << description;
  } else if (!device.empty()) {
    label << device;
  } else {
    label << "adapter";
  }
  if (!vendor.empty()) {
    label << " vendor=" << vendor;
  }
  label << " vendorID=0x" << std::hex << info.vendorID << " deviceID=0x" << info.deviceID;
  auto result = label.str();
  wgpuAdapterInfoFreeMembers(info);
  return result;
}

}  // namespace

auto WebGpuContext::Instance() -> WebGpuContext& {
  static WebGpuContext context;
  return context;
}

WebGpuContext::WebGpuContext() {
  const wgpu::InstanceFeatureName instance_features[] = {wgpu::InstanceFeatureName::TimedWaitAny};
  wgpu::InstanceLimits            instance_limits{};
  instance_limits.timedWaitAnyMaxCount = 1;
  wgpu::InstanceDescriptor instance_descriptor{};
  instance_descriptor.requiredFeatureCount = 1;
  instance_descriptor.requiredFeatures     = instance_features;
  instance_descriptor.requiredLimits       = &instance_limits;

#ifdef ALCEDO_DAWN_D3D_RUNTIME_DIR
  const char*                          runtime_search_paths[] = {ALCEDO_DAWN_D3D_RUNTIME_DIR};
  dawn::native::DawnInstanceDescriptor dawn_descriptor{};
  dawn_descriptor.additionalRuntimeSearchPathsCount = 1;
  dawn_descriptor.additionalRuntimeSearchPaths      = runtime_search_paths;
  instance_descriptor.nextInChain                   = &dawn_descriptor;
#endif

  native_instance_ = std::make_unique<dawn::native::Instance>(&instance_descriptor);

  const std::vector<AdapterAttempt> attempts = {
#ifdef _WIN32
      {wgpu::BackendType::D3D12, wgpu::FeatureLevel::Core, "D3D12/Core"},
      {wgpu::BackendType::D3D12, wgpu::FeatureLevel::Compatibility, "D3D12/Compatibility"},
#endif
  };

  std::ostringstream log;
  for (const auto& attempt : attempts) {
    wgpu::RequestAdapterOptions options{};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    options.backendType     = attempt.backend_type;
    options.featureLevel    = attempt.feature_level;

    auto adapters           = native_instance_->EnumerateAdapters(&options);
    log << attempt.name << ": " << adapters.size() << " adapter(s)";
    if (adapters.empty()) {
      log << '\n';
      continue;
    }

    for (auto& adapter : adapters) {
      log << "\n  trying " << AdapterInfoLabel(adapter.Get());
      wgpu::DeviceDescriptor device_descriptor{};
      device_descriptor.SetDeviceLostCallback(
          wgpu::CallbackMode::AllowProcessEvents,
          [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView) {});
      WGPUDevice raw_device = adapter.CreateDevice(&device_descriptor);
      if (raw_device == nullptr) {
        log << ": CreateDevice failed";
        continue;
      }

      device_    = wgpu::Device::Acquire(raw_device);
      queue_     = device_.GetQueue();
      available_ = true;
      log << ": selected";
      initialization_log_ = log.str();
      return;
    }
    log << '\n';
  }
  initialization_log_ = log.str();
}

WebGpuContext::~WebGpuContext() = default;

auto WebGpuContext::IsAvailable() const noexcept -> bool { return available_ && device_.Get(); }

auto WebGpuContext::InitializationLog() const noexcept -> const std::string& {
  return initialization_log_;
}

auto WebGpuContext::Device() const -> const wgpu::Device& {
  if (!IsAvailable()) {
    throw std::runtime_error("WebGpuContext: WebGPU device is unavailable.");
  }
  return device_;
}

auto WebGpuContext::Queue() const -> const wgpu::Queue& {
  if (!IsAvailable() || !queue_.Get()) {
    throw std::runtime_error("WebGpuContext: WebGPU queue is unavailable.");
  }
  return queue_;
}

void WebGpuContext::Wait(const wgpu::Future& future) const {
  if (!native_instance_) {
    throw std::runtime_error("WebGpuContext: WebGPU instance is unavailable.");
  }
  wgpu::FutureWaitInfo wait_info{future};
  const auto           status = static_cast<wgpu::WaitStatus>(wgpuInstanceWaitAny(
      native_instance_->Get(), 1, reinterpret_cast<WGPUFutureWaitInfo*>(&wait_info),
      std::numeric_limits<uint64_t>::max()));
  if (status != wgpu::WaitStatus::Success) {
    throw std::runtime_error("WebGpuContext: WebGPU wait failed.");
  }
}

void WebGpuContext::WaitForSubmittedWork() const {
  bool done   = false;
  auto future = Queue().OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::QueueWorkDoneStatus status, wgpu::StringView, bool* complete) {
        if (status == wgpu::QueueWorkDoneStatus::Success) {
          *complete = true;
        }
      },
      &done);
  Wait(future);
  if (!done) {
    throw std::runtime_error("WebGpuContext: WebGPU submitted work did not complete.");
  }
}

}  // namespace webgpu
}  // namespace alcedo

#endif
