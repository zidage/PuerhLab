//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "utils/cuda/cuda_driver_requirements.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include <sstream>
#include <utility>

namespace {

constexpr int kCudaSuccess = 0;

#if defined(_WIN32)
using CuInitFn = int(__stdcall*)(unsigned int);
using CuDriverGetVersionFn = int(__stdcall*)(int*);

class ScopedModule final {
 public:
  explicit ScopedModule(HMODULE module) : module_(module) {}
  ScopedModule(const ScopedModule&) = delete;
  auto operator=(const ScopedModule&) -> ScopedModule& = delete;

  ScopedModule(ScopedModule&& other) noexcept : module_(std::exchange(other.module_, nullptr)) {}

  auto operator=(ScopedModule&& other) noexcept -> ScopedModule& {
    if (this != &other) {
      Reset();
      module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
  }

  ~ScopedModule() { Reset(); }

  [[nodiscard]] auto Get() const -> HMODULE { return module_; }

 private:
  void Reset() {
    if (module_ != nullptr) {
      FreeLibrary(module_);
      module_ = nullptr;
    }
  }

  HMODULE module_ = nullptr;
};
#endif

}  // namespace

namespace puerhlab::cuda {

auto IsCudaDriverVersionSupported(int detected_cuda_driver_version,
                                  int minimum_cuda_driver_version) -> bool {
  return detected_cuda_driver_version >= minimum_cuda_driver_version;
}

auto FormatCudaVersion(int cuda_driver_version) -> std::string {
  if (cuda_driver_version <= 0) {
    return "unknown";
  }

  const int major = cuda_driver_version / 1000;
  const int minor = (cuda_driver_version % 1000) / 10;

  std::ostringstream oss;
  oss << major << '.' << minor;
  return oss.str();
}

auto CheckDriverSupport(int minimum_cuda_driver_version) -> DriverSupportInfo {
#if !defined(_WIN32)
  (void)minimum_cuda_driver_version;
  return {
      .status = DriverSupportStatus::kSupported,
      .detected_cuda_driver_version = 0,
      .detail = {},
  };
#else
  ScopedModule cuda_driver(::LoadLibraryW(L"nvcuda.dll"));
  if (cuda_driver.Get() == nullptr) {
    return {
        .status = DriverSupportStatus::kDriverUnavailable,
        .detected_cuda_driver_version = 0,
        .detail = "nvcuda.dll was not found.",
    };
  }

  const auto cu_init = reinterpret_cast<CuInitFn>(::GetProcAddress(cuda_driver.Get(), "cuInit"));
  const auto cu_driver_get_version = reinterpret_cast<CuDriverGetVersionFn>(
      ::GetProcAddress(cuda_driver.Get(), "cuDriverGetVersion"));
  if (cu_init == nullptr || cu_driver_get_version == nullptr) {
    return {
        .status = DriverSupportStatus::kQueryFailed,
        .detected_cuda_driver_version = 0,
        .detail = "Failed to resolve CUDA driver entry points from nvcuda.dll.",
    };
  }

  const int init_status = cu_init(0);
  if (init_status != kCudaSuccess) {
    return {
        .status = DriverSupportStatus::kDriverUnavailable,
        .detected_cuda_driver_version = 0,
        .detail = "cuInit failed with error code " + std::to_string(init_status) + '.',
    };
  }

  int detected_cuda_driver_version = 0;
  const int version_status = cu_driver_get_version(&detected_cuda_driver_version);
  if (version_status != kCudaSuccess) {
    return {
        .status = DriverSupportStatus::kQueryFailed,
        .detected_cuda_driver_version = 0,
        .detail = "cuDriverGetVersion failed with error code " + std::to_string(version_status) + '.',
    };
  }

  if (!IsCudaDriverVersionSupported(detected_cuda_driver_version, minimum_cuda_driver_version)) {
    return {
        .status = DriverSupportStatus::kDriverTooOld,
        .detected_cuda_driver_version = detected_cuda_driver_version,
        .detail = {},
    };
  }

  return {
      .status = DriverSupportStatus::kSupported,
      .detected_cuda_driver_version = detected_cuda_driver_version,
      .detail = {},
  };
#endif
}

}  // namespace puerhlab::cuda
