//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "storage/service/sleeve/element/element_id_service.hpp"

#include <utility>

#include "utils/string/convert.hpp"

namespace alcedo {
auto ElementIdService::ToParams(const sl_element_id_t& source) -> ElementIdMapperParams {
  return {source};
}

auto ElementIdService::FromParams(ElementIdMapperParams&& param) -> sl_element_id_t {
  return param.id;
}

auto ElementIdService::GetElementIdsByQuery(const std::wstring& query_sql)
    -> std::vector<sl_element_id_t> {
  std::string query_sql_u8 = conv::ToBytes(query_sql);
  return GetByQuery(std::move(query_sql_u8));
}
}  // namespace alcedo
