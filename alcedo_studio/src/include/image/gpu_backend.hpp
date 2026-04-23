//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.
#pragma once

namespace alcedo {

enum class GpuBackendKind {
  None,
  CUDA,
  Metal,
  WebGPU,
};

}  // namespace alcedo
