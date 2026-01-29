//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "storage/service/sleeve/element/element_id_service.hpp"

#include <utility>

#include "utils/string/convert.hpp"

namespace puerhlab {
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
}  // namespace puerhlab
