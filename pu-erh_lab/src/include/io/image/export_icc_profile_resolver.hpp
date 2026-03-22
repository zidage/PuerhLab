//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "io/image/export_color_profile_config.hpp"

namespace puerhlab {

class ExportIccProfileResolver {
 public:
  static auto ResolveIccProfileBytes(const ExportColorProfileConfig& config) -> std::vector<uint8_t>;
  static auto ResolveConfigProfilePath(const ExportColorProfileConfig& config)
      -> std::optional<std::filesystem::path>;
};

}  // namespace puerhlab
