#pragma once

#include <chrono>

namespace puerhlab::ui::controllers::render {

inline constexpr std::chrono::milliseconds kFastPreviewMinSubmitInterval{16};
inline constexpr std::chrono::milliseconds kQualityPreviewDebounceInterval{180};

auto CanSubmitFastPreviewNow(
    const std::chrono::steady_clock::time_point& last_submit_time,
    const std::chrono::steady_clock::time_point& now) -> bool;

auto ComputeFastPreviewDelayMs(
    const std::chrono::steady_clock::time_point& last_submit_time,
    const std::chrono::steady_clock::time_point& now) -> int;

}  // namespace puerhlab::ui::controllers::render
