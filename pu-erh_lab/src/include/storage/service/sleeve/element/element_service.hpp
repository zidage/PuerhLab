//  Copyright 2025 Yurun Zi
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

#include <codecvt>
#include <filesystem>
#include <memory>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "storage/mapper/sleeve/element/element_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ElementService
    : public ServiceInterface<ElementService, std::shared_ptr<SleeveElement>, ElementMapperParams,
                              ElementMapper, sl_element_id_t> {
 private:
 public:
  using ServiceInterface::ServiceInterface;
  static auto ToParams(const std::shared_ptr<SleeveElement>& source) -> ElementMapperParams;
  static auto FromParams(ElementMapperParams&& param) -> std::shared_ptr<SleeveElement>;

  auto        GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
  auto GetElementByName(const std::wstring& name) -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementByType(const ElementType type) -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementsInFolderByFilter(const std::wstring& filter_sql)
      -> std::vector<std::shared_ptr<SleeveElement>>;
};
};  // namespace puerhlab