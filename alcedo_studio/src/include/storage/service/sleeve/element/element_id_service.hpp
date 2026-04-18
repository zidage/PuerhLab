//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <duckdb.h>

#include <string>
#include <vector>

#include "storage/mapper/sleeve/element/element_id_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace alcedo {
class ElementIdService
    : public ServiceInterface<ElementIdService, sl_element_id_t, ElementIdMapperParams,
                              ElementIdMapper, sl_element_id_t> {
 public:
  using ServiceInterface::ServiceInterface;

  static auto ToParams(const sl_element_id_t& source) -> ElementIdMapperParams;
  static auto FromParams(ElementIdMapperParams&& param) -> sl_element_id_t;

  // For specialized queries only: expects the query to return a single UINT32 column named "id".
  auto GetElementIdsByQuery(const std::wstring& query_sql) -> std::vector<sl_element_id_t>;
};
};  // namespace alcedo
