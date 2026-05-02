//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

namespace alcedo::ui {

template <typename T>
struct PipelineLoadResult {
  T state;
  bool loaded_any = false;
};

}  // namespace alcedo::ui
