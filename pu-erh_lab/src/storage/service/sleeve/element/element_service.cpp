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

#include "storage/service/sleeve/element/element_service.hpp"

#include <cstdint>
#include <ctime>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
auto ElementService::ToParams(const std::shared_ptr<SleeveElement>& source) -> ElementMapperParams {
  char added_time[32];
  char modified_time[32];
  std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S",
                std::gmtime(&source->added_time_));
  std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S",
                std::gmtime(&source->last_modified_time_));

  std::string utf_8_str = conv::ToBytes(source->element_name_);
  return {source->element_id_,
          static_cast<uint32_t>(source->type_),
          std::make_unique<std::string>(utf_8_str),
          std::make_unique<std::string>(added_time),
          std::make_unique<std::string>(modified_time),
          source->ref_count_};
}

auto ElementService::FromParams(ElementMapperParams&& param) -> std::shared_ptr<SleeveElement> {
  auto               id                = param.id_;
  auto               type              = static_cast<ElementType>(param.type_);
  auto               element_name      = conv::FromBytes(std::move(*param.element_name_));
  auto               added_time_str    = std::move(*param.added_time_);
  auto               modified_time_str = std::move(*param.modified_time_);
  auto               ref_count         = param.ref_count_;

  std::tm            tm_added{};
  std::tm            tm_modified{};

  std::istringstream a_ss(added_time_str);
  std::istringstream m_ss(modified_time_str);
  a_ss >> std::get_time(&tm_added, "%Y-%m-%d %H:%M:%S");
  m_ss >> std::get_time(&tm_modified, "%Y-%m-%d %H:%M:%S");
  std::shared_ptr<SleeveElement> element;
  if (type == ElementType::FILE) {
    element = std::make_shared<SleeveFile>(id, element_name);
  } else if (type == ElementType::FOLDER) {
    element = std::make_shared<SleeveFolder>(id, element_name);
  } else {
    throw std::runtime_error("ElementService: Invalid ElementMapperParams");
  }

  element->added_time_         = std::mktime(&tm_added);
  element->last_modified_time_ = std::mktime(&tm_modified);
  element->ref_count_          = ref_count;

  return element;
}

auto ElementService::GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  auto result = GetByPredicate(std::format("id={}", id));
  if (result.size() > 1) {
    throw std::runtime_error("Element Service: Sleeve element id is not unique. Broken DB file");
  }
  if (result.size() == 0) {
    throw std::runtime_error(
        std::format("Element Service: No element with id {} is stored in DB", id));
  }
  return result[0];
}

auto ElementService::GetElementByName(const std::wstring& name)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  std::string predicate = conv::ToBytes(std::format(L"element_name={}", name));
  return GetByPredicate(std::move(predicate));
}

auto ElementService::GetElementByType(const ElementType type)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  std::string predicate = std::format("type={}", static_cast<uint32_t>(type));
  return GetByPredicate(std::move(predicate));
}

auto ElementService::GetElementsInFolderByFilter(const std::wstring& filter_sql)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  std::string filter_sql_u8 = conv::ToBytes(filter_sql);
  return GetByQuery(std::move(filter_sql_u8));
}
};  // namespace puerhlab