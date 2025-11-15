/*
 * @file        pu-erh_lab/src/include/mapper/sleeve/sleeve_filter/filter_combo.hpp
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

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "filters/sleeve_filter.hpp"
#include "image/image.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class FilterOp { AND, OR, NOT };

enum class FilterField {
  ExifCameraModel,
  ExifFocalLength,
  ExifAperture,
  ExifISO,
  CaptureDate,
  ImportDate,
  FileName,
  FileExtension,
  ImageSize,
  Rating,
  ImagePath,
  SemanticTags
};

enum class CompareOp {
  EQUALS,
  NOT_EQUALS,
  CONTAINS,
  NOT_CONTAINS,
  GREATER_THAN,
  LESS_THAN,
  GREATER_EQUAL,
  LESS_EQUAL,
  STARTS_WITH,
  ENDS_WITH,
  BETWEEN,
  REGEX
};

using FilterValue = std::variant<std::monostate, int64_t, double, bool, std::wstring, std::tm>;

struct FieldCondition {
  FilterField                field;
  CompareOp                  op;
  FilterValue                value;
  std::optional<FilterValue> second_value = std::nullopt;  // Used for BETWEEN condition
};

struct FilterNode {
  enum class Type { Logical, Condition, RawSQL } type;

  // For Logical nodes
  FilterOp                      op;
  std::vector<FilterNode>       children;

  // For Condition nodes
  std::optional<FieldCondition> condition;

  // For RawSQL nodes
  std::optional<std::wstring>   raw_sql;
};

class FilterSQLCompiler {
 public:
  struct Result {
    std::wstring             where_clause;
    std::vector<FilterValue> params;
  };

  static Result Compile(const FilterNode& node);

 private:
  static std::wstring CompileNode(const FilterNode& node);
  static std::wstring FieldToColumn(FilterField field);
  static std::wstring CompareToSQL(CompareOp op);
};

class FilterCombo {
 private:
  FilterNode                _root;
  std::vector<SleeveFilter> _filters;

 public:
  filter_id_t filter_id;
};
};  // namespace puerhlab