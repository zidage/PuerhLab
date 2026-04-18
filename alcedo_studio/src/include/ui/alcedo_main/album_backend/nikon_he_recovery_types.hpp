//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QString>

#include <filesystem>
#include <vector>

#include "type/type.hpp"

namespace alcedo::ui {

enum class NikonHeRecoveryPhase : uint8_t {
  IDLE = 0,
  REVIEW_UNSUPPORTED,
  SELECTING_CONVERTER,
  RUNNING_CONVERTER,
  VALIDATING_DNG,
  REMOVING_PROJECT_ITEMS,
  REIMPORTING_DNG,
  FINISHED,
  ABORTED,
};

struct NikonHeRecoveryItem {
  sl_element_id_t       element_id_ = 0;
  image_id_t            image_id_   = 0;
  file_name_t           file_name_{};
  std::filesystem::path source_path_{};
  std::filesystem::path converted_dng_path_{};
};

using NikonHeRecoveryItemList = std::vector<NikonHeRecoveryItem>;

}  // namespace alcedo::ui
