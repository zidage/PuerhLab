//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <libraw/libraw.h>

#if !defined(_WIN32)
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#endif

namespace alcedo {
namespace libraw_guard {

inline constexpr bool kAsanEnabled =
#if defined(__has_feature)
    __has_feature(address_sanitizer) ||
#endif
#if defined(__SANITIZE_ADDRESS__)
    true;
#else
    false;
#endif

inline constexpr bool kNeedOpenMpUnpackGuard =
#if defined(_WIN32)
    false;
#elif defined(__APPLE__)
    true;
#else
    kAsanEnabled;
#endif

inline void ConfigureOpenMpEnvironment() {
#if defined(_WIN32)
  return;
#else
  if constexpr (!kNeedOpenMpUnpackGuard) {
    return;
  }

  static const bool configured = []() {
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OMP_THREAD_LIMIT", "1", 1);
    setenv("OMP_DYNAMIC", "FALSE", 1);
    setenv("OMP_NESTED", "FALSE", 1);
    setenv("OMP_MAX_ACTIVE_LEVELS", "1", 1);
    setenv("KMP_ALL_THREADS", "1", 1);
    return true;
  }();
  (void)configured;
#endif
}

inline void ConfigureOpenMpRuntime() {
#if defined(_WIN32)
  return;
#else
  if constexpr (!kNeedOpenMpUnpackGuard) {
    return;
  }

  ConfigureOpenMpEnvironment();

  using OmpSetIntFn = void (*)(int);
  auto* const global_symbols = RTLD_DEFAULT;

  if (const auto set_dynamic =
          reinterpret_cast<OmpSetIntFn>(dlsym(global_symbols, "omp_set_dynamic"));
      set_dynamic != nullptr) {
    set_dynamic(0);
  }
  if (const auto set_nested =
          reinterpret_cast<OmpSetIntFn>(dlsym(global_symbols, "omp_set_nested"));
      set_nested != nullptr) {
    set_nested(0);
  }
  if (const auto set_max_active_levels =
          reinterpret_cast<OmpSetIntFn>(dlsym(global_symbols, "omp_set_max_active_levels"));
      set_max_active_levels != nullptr) {
    set_max_active_levels(1);
  }
  if (const auto set_num_threads =
          reinterpret_cast<OmpSetIntFn>(dlsym(global_symbols, "omp_set_num_threads"));
      set_num_threads != nullptr) {
    set_num_threads(1);
  }
#endif
}

inline auto Unpack(LibRaw& raw_processor) -> int {
#if defined(_WIN32)
  return raw_processor.unpack();
#else
  if constexpr (!kNeedOpenMpUnpackGuard) {
    return raw_processor.unpack();
  }

  static std::mutex unpack_mutex;
  std::lock_guard<std::mutex> lock(unpack_mutex);
  ConfigureOpenMpRuntime();
  return raw_processor.unpack();
#endif
}

}  // namespace libraw_guard
}  // namespace alcedo
