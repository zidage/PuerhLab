#include "ui/puerhlab_main/editor_dialog/controllers/render_controller.hpp"

namespace puerhlab::ui::controllers::render {

auto CanSubmitFastPreviewNow(
    const std::chrono::steady_clock::time_point& last_submit_time,
    const std::chrono::steady_clock::time_point& now) -> bool {
  if (last_submit_time.time_since_epoch().count() == 0) {
    return true;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_submit_time);
  return elapsed >= kFastPreviewMinSubmitInterval;
}

auto ComputeFastPreviewDelayMs(
    const std::chrono::steady_clock::time_point& last_submit_time,
    const std::chrono::steady_clock::time_point& now) -> int {
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_submit_time);
  auto       delay   = kFastPreviewMinSubmitInterval - elapsed;
  if (delay <= std::chrono::milliseconds::zero()) {
    delay = std::chrono::milliseconds{1};
  }
  return static_cast<int>(delay.count());
}

}  // namespace puerhlab::ui::controllers::render
