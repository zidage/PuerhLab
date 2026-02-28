#pragma once

#include <chrono>

namespace puerhlab::ui {

struct EditorDialogTiming {
  std::chrono::milliseconds fast_preview_min_submit_interval_{16};
  std::chrono::milliseconds quality_preview_debounce_interval_{180};
};

}  // namespace puerhlab::ui
