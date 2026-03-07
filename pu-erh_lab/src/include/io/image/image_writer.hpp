//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <memory>

#include "image/image_buffer.hpp"
#include "type/supported_file_type.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImageWriter {
 public:
  static void WriteImageToPath(const image_path_t&          src_path,
                               std::shared_ptr<ImageBuffer> image_data,
                               ExportFormatOptions          options);
};
};  // namespace puerhlab