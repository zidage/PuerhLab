//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#ifdef HAVE_METAL
#include "metal/metal_context.hpp"
namespace puerhlab {
MetalContext::MetalContext() {
  device_ = MTL::CreateSystemDefaultDevice();
  if (!device_) {
    throw std::runtime_error("[FATAL] MetalContext: Failed to create Metal device.");
  }
  queue_ = device_->newCommandQueue();
  if (!queue_) {
    throw std::runtime_error("[FATAL] MetalContext: Failed to create Metal command queue.");
  }
}

MetalContext::~MetalContext() {
  if (queue_) {
    queue_->release();
    queue_ = nullptr;
  }
  if (device_) {
    device_->release();
    device_ = nullptr;
  }
}

auto MetalContext::Instance() -> MetalContext& {
  static MetalContext instance;
  return instance;
}

auto MetalContext::Device() const -> MTL::Device* { return device_; }

auto MetalContext::Queue() const -> MTL::CommandQueue* { return queue_; }
}  // namespace puerhlab
#endif
