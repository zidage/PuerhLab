//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/mapper/sleeve/element/element_id_mapper.hpp"

#include <stdexcept>

namespace alcedo {
auto ElementIdMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementIdMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for ElementIdMapper");
  }

  auto id = std::get_if<sl_element_id_t>(&data[0]);
  if (id == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }

  return {*id};
}
}  // namespace alcedo
