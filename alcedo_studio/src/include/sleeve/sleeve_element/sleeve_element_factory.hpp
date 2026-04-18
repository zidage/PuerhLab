//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>

#include "sleeve_element.hpp"
#include "sleeve_file.hpp"
#include "sleeve_folder.hpp"

namespace alcedo {
class SleeveElementFactory {
 public:
  static std::shared_ptr<SleeveElement> CreateElement(const ElementType& type, sl_element_id_t id,
                                                      file_name_t element_name);
};
};  // namespace alcedo