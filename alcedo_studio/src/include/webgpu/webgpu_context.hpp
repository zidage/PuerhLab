//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.
#pragma once

#ifdef HAVE_WEBGPU

#include <webgpu/webgpu_cpp.h>

#include <memory>
#include <string>

namespace dawn::native {
class Instance;
}  // namespace dawn::native

namespace alcedo {
namespace webgpu {

class WebGpuContext {
 public:
  static auto        Instance() -> WebGpuContext&;

  [[nodiscard]] auto IsAvailable() const noexcept -> bool;
  [[nodiscard]] auto InitializationLog() const noexcept -> const std::string&;
  [[nodiscard]] auto Device() const -> const wgpu::Device&;
  [[nodiscard]] auto Queue() const -> const wgpu::Queue&;
  void               Wait(const wgpu::Future& future) const;
  void               WaitForSubmittedWork() const;

 private:
  WebGpuContext();
  ~WebGpuContext();

  WebGpuContext(const WebGpuContext&)                    = delete;
  auto operator=(const WebGpuContext&) -> WebGpuContext& = delete;

  std::unique_ptr<dawn::native::Instance> native_instance_;
  wgpu::Device                            device_ = nullptr;
  wgpu::Queue                             queue_  = nullptr;
  std::string                             initialization_log_;
  bool                                    available_ = false;
};

}  // namespace webgpu
}  // namespace alcedo

#endif
