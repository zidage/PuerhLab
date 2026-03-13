//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "edit/operators/utils/color_utils.hpp"
#include "image/image_buffer.hpp"
#include "type/supported_file_type.hpp"
#include "type/type.hpp"

namespace puerhlab {
struct ExportColorProfileConfig {
  ColorUtils::ColorSpace encoding_space = ColorUtils::ColorSpace::REC709;
  ColorUtils::EOTF       encoding_eotf  = ColorUtils::EOTF::GAMMA_2_2;
  float                  peak_luminance = 100.0f;
};

class ImageWriter {
 public:
  static auto ShouldWriteUltraHdr(
      const ExportFormatOptions&                          options,
      const std::optional<ExportColorProfileConfig>& color_profile) -> bool;

  static void WriteImageToPath(const image_path_t&          src_path,
                               std::shared_ptr<ImageBuffer> image_data,
                               ExportFormatOptions          options,
                               std::optional<ExportColorProfileConfig> color_profile = std::nullopt);
};
};  // namespace puerhlab
