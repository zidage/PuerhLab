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

#pragma once

#include <duckdb.h>

#include <string>
#include <vector>

#include "storage/mapper/sleeve/element/element_id_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
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
};  // namespace puerhlab
