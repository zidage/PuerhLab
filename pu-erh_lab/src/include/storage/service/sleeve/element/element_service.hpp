//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

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