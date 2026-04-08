//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <string>

namespace puerhlab::cuda {

inline constexpr int kMinimumSupportedCudaDriverVersion = 12080;

enum class DriverSupportStatus {
  kSupported,
  kDriverUnavailable,
  kDriverTooOld,
  kQueryFailed,
};

struct DriverSupportInfo {
  DriverSupportStatus status = DriverSupportStatus::kQueryFailed;
  int                 detected_cuda_driver_version = 0;
  std::string         detail;

  [[nodiscard]] auto IsSupported() const -> bool {
    return status == DriverSupportStatus::kSupported;
  }
};

auto IsCudaDriverVersionSupported(
    int detected_cuda_driver_version,
    int minimum_cuda_driver_version = kMinimumSupportedCudaDriverVersion) -> bool;

auto FormatCudaVersion(int cuda_driver_version) -> std::string;

auto CheckDriverSupport(
    int minimum_cuda_driver_version = kMinimumSupportedCudaDriverVersion) -> DriverSupportInfo;

}  // namespace puerhlab::cuda
