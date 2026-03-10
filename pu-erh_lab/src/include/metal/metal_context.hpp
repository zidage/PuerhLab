//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_METAL
#include <puerhlab/metal/Metal.hpp>
namespace puerhlab {
class MetalContext {
 private:
  MTL::Device*       device_ = nullptr;
  MTL::CommandQueue* queue_ = nullptr;

  MetalContext();

  ~MetalContext();

 public:
  MetalContext(const MetalContext&)                    = delete;
  auto operator=(const MetalContext&) -> MetalContext& = delete;
  MetalContext(MetalContext&&)                         = delete;
  auto operator=(MetalContext&&) -> MetalContext&      = delete;

  static auto Instance() -> MetalContext&;

  auto        Device() const -> MTL::Device*;
  auto Queue() const -> MTL::CommandQueue*;
};
};


#endif
