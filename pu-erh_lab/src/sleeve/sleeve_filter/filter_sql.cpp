/*
 * @file        pu-erh_lab/src/sleeve/sleeve_filter/filter_sql.cpp
 * @brief       A combination of a set of filters
 * @author      Yurun Zi
 * @date        2025-11-14
 * @license     GPL-3.0-or-later
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (C) 2025  Yurun Zi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <sstream>
#include <string>

#include "sleeve/sleeve_filter/filter_combo.hpp"

namespace puerhlab {
std::wstring FilterSQLCompiler::FieldToColumn(FilterField field) {
  switch (field) {
    case FilterField::ExifCameraModel:
      return L"json_extract(metadata, '$.Model')";
    case FilterField::ExifFocalLength:
      return L"json_extract(metadata, '$.FocalLength')::DOUBLE";
    case FilterField::ExifAperture:
      return L"json_extract(metadata, '$.Aperture')::DOUBLE";
    case FilterField::ExifISO:
      return L"json_extract(metadata, '$.ISO')::INT";
    case FilterField::CaptureDate:
      return L"json_extract(metadata, '$.DateTimeString')::TIMESTAMP";
    case FilterField::ImportDate:
      return L"added_time";
    case FilterField::FileName:
      return L"element_name";
    case FilterField::FileExtension:
      return L"UPPER(file_name)";  // Avoid case sensitivity issues
    case FilterField::ImageSize:
      return L"json_extract(metadata, '$.ImageSize')";
    case FilterField::Rating:
      return L"json_extract(metadata, '$.Rating')";
    case FilterField::ImagePath:
      return L"image_path";
    case FilterField::SemanticTags:
      return L"embedding";  // Needs further processing
      break;
  }
  return L"";
}

std::wstring FilterSQLCompiler::CompareToSQL(CompareOp op) {
  switch (op) {
    case CompareOp::EQUALS:
      return L"=";
    case CompareOp::NOT_EQUALS:
      return L"!=";
    case CompareOp::CONTAINS:
      return L"LIKE";
    case CompareOp::NOT_CONTAINS:
      return L"NOT LIKE";
    case CompareOp::GREATER_THAN:
      return L">";
    case CompareOp::LESS_THAN:
      return L"<";
    case CompareOp::GREATER_EQUAL:
      return L">=";
    case CompareOp::LESS_EQUAL:
      return L"<=";
    case CompareOp::STARTS_WITH:
      return L"LIKE";
    case CompareOp::ENDS_WITH:
      return L"LIKE";
    case CompareOp::BETWEEN:
      return L"BETWEEN";
    case CompareOp::REGEX:
      return L"REGEXP";
    default:
      break;
  }
  return L"";
}

static inline std::wstring FilterValueToString(const FilterValue& value) {
  if (std::holds_alternative<std::monostate>(value)) {
    return L"NULL";
  } else if (std::holds_alternative<int64_t>(value)) {
    return std::to_wstring(std::get<int64_t>(value));
  } else if (std::holds_alternative<double>(value)) {
    return std::to_wstring(std::get<double>(value));
  } else if (std::holds_alternative<bool>(value)) {
    return std::get<bool>(value) ? L"1" : L"0";
  } else if (std::holds_alternative<std::wstring>(value)) {
    return L"'" + std::get<std::wstring>(value) + L"'";
  } else if (std::holds_alternative<std::tm>(value)) {
    const std::tm& tm_value = std::get<std::tm>(value);
    wchar_t buffer[20];
    wcsftime(buffer, sizeof(buffer), L"'%Y-%m-%d %H:%M:%S'", &tm_value);
    return L"TIMESTAMP " + std::wstring(buffer);
  }
  return L"";
}

std::wstring FilterSQLCompiler::GenerateConditionString(const FieldCondition& cond) {
  std::wstring column = FilterSQLCompiler::FieldToColumn(cond.field);
  std::wstring op = FilterSQLCompiler::CompareToSQL(cond.op);
  auto& value = cond.value;

  // First four cases does not use op string directly
  if (cond.op == CompareOp::BETWEEN && cond.second_value.has_value()) {
    return std::format(L"({} BETWEEN {} AND {})", column, FilterValueToString(value), FilterValueToString(cond.second_value.value()));
  } else if (cond.op == CompareOp::CONTAINS) {
    return std::format(L"({} LIKE '%{}%')", column, std::get<std::wstring>(value));
  } else if (cond.op == CompareOp::STARTS_WITH) {
    return std::format(L"({} LIKE '{}%')", column, std::get<std::wstring>(value));
  } else if (cond.op == CompareOp::ENDS_WITH) {
    return std::format(L"({} LIKE '%{}')", column, std::get<std::wstring>(value));
  } else {
    return std::format(L"({} {} {})", column, op, FilterValueToString(value));
  }
}

std::wstring FilterSQLCompiler::CompileNode(const FilterNode& node) {
  if (node.type == FilterNode::Type::Condition && node.condition.has_value()) {
    const FieldCondition& cond = node.condition.value();
    return GenerateConditionString(cond);
  } else if (node.type == FilterNode::Type::Logical) {
    std::wstring combined;
    // Reserve some space to avoid multiple allocations
    combined.reserve(256);
    combined.append(L"(");
    for (size_t i = 0; i < node.children.size(); ++i) {
      if (i > 0) {
        // TODO: Add handling for NOT
        combined.append((node.op == FilterOp::AND) ? L" AND " : L" OR ");
      }
      combined.append(CompileNode(node.children[i]));
    }
    combined.append(L")");
    return combined;
  } else if (node.type == FilterNode::Type::RawSQL && node.raw_sql.has_value()) {
    return node.raw_sql.value();
  }
  return L"";
}

std::wstring FilterSQLCompiler::Compile(const FilterNode& node) {
  return CompileNode(node);
}


};  // namespace puerhlab