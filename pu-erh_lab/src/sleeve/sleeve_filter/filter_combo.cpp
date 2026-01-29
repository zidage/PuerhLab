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

#include "sleeve/sleeve_filter/filter_combo.hpp"

#include <cstddef>
#include <set>
#include <unordered_map>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FilterCombo::GenerateSQLOn(sl_element_id_t parent_id) const -> std::wstring {
  // For now, we just compile the filter node to SQL
  std::wstring sql = std::format(
      L"SELECT\n"
      L"  e.*\n"
      L"FROM\n"
      L"  FolderContent fc\n"
      L"JOIN Element e ON e.id = fc.element_id\n"
      L"JOIN FileImage fi ON fi.file_id = e.id\n"
      L"JOIN Image img ON img.id = fi.image_id\n"
      L"WHERE\n"
      L"  fc.folder_id = {}\n"
      L"  AND e.type = {}\n"
      L"  AND ({});",
      parent_id, static_cast<uint32_t>(ElementType::FILE), FilterSQLCompiler::Compile(root_));
  return sql;
}

auto FilterCombo::GenerateIdSQLOn(sl_element_id_t parent_id) const -> std::wstring {
  // For now, we just compile the filter node to SQL
  std::wstring sql = std::format(
      L"SELECT\n"
      L"  e.id\n"
      L"FROM\n"
      L"  FolderContent fc\n"
      L"JOIN Element e ON e.id = fc.element_id\n"
      L"JOIN FileImage fi ON fi.file_id = e.id\n"
      L"JOIN Image img ON img.id = fi.image_id\n"
      L"WHERE\n"
      L"  fc.folder_id = {}\n"
      L"  AND e.type = {}\n"
      L"  AND ({});",
      parent_id, static_cast<uint32_t>(ElementType::FILE), FilterSQLCompiler::Compile(root_));
  return sql;
}
};  // namespace puerhlab