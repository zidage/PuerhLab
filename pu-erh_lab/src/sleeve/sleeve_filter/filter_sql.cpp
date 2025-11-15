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

#include <string>

#include "sleeve/sleeve_filter/filter_combo.hpp"

namespace puerhlab {
std::wstring FilterSQLCompiler::FieldToColumn(FilterField field) {
  switch (field) {
    case FilterField::ExifCameraModel:
      return L"json_extract_scalar(metadata, '$.Model')";
    case FilterField::ExifFocalLength:
      return L"json_extract_scalar(metadata, '$.FocalLength')";
    case FilterField::ExifAperture:
      return L"json_extract_scalar(metadata, '$.Aperture')";
    case FilterField::ExifISO:
      return L"json_extract_scalar(metadata, '$.ISO')";
    case FilterField::CaptureDate:
      return L"json_extract_scalar(metadata, '$.DateTimeString')";
    case FilterField::ImportDate:
      return L"added_time";
    case FilterField::FileName:
      return L"element_name";
    case FilterField::FileExtension:
      return L"file_name";  // Needs further processing
    case FilterField::ImageSize:
      return L"json_extract_scalar(metadata, '$.ImageSize')";
    case FilterField::Rating:
      return L"json_extract_scalar(metadata, '$.Rating')";
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

FilterSQLCompiler::Result FilterSQLCompiler::Compile(const FilterNode& node) {
  Result result;

  // Implementation of the SQL compilation logic goes here.
  // This is a placeholder implementation.

  // Traverse the AST and build the SQL query
  result.where_clause = CompileNode(node);

  return result;
}

std::wstring FilterSQLCompiler::CompileNode(const FilterNode& node) {
  if (node.type == FilterNode::Type::Condition && node.condition.has_value()) {
    const FieldCondition& cond = node.condition.value();
    std::wstring column = FieldToColumn(cond.field);
    std::wstring op = CompareToSQL(cond.op);
    // Here we would also need to handle the value and second_value properly
    // TODO: Add special handling for different data types and operators
    return L"(" + column + L" " + op + L" ?)";
  } else if (node.type == FilterNode::Type::Logical) {
    std::wstring combined;
    for (size_t i = 0; i < node.children.size(); ++i) {
      if (i > 0) {
        // TODO: Add handling for NOT
        combined += (node.op == FilterOp::AND) ? L" AND " : L" OR ";
      }
      combined += CompileNode(node.children[i]);
    }
    return L"(" + combined + L")";
  } else if (node.type == FilterNode::Type::RawSQL && node.raw_sql.has_value()) {
    return node.raw_sql.value();
  }
  return L"";
}


};  // namespace puerhlab