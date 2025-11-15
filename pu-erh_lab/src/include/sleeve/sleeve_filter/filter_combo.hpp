/*
 * @file        pu-erh_lab/src/include/mapper/sleeve/sleeve_filter/filter_combo.hpp
 * @brief       A combination of a set of filters
 * @author      Yurun Zi
 * @date        2025-03-26
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "filters/sleeve_filter.hpp"
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
  FileSize,
  Rating,          
  FolderPath,
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

using FilterValue =
    std::variant<std::monostate, int64_t, double, bool, std::wstring, std::tm>;

struct FieldCondition {
  FilterField field;
  CompareOp  op;
  FilterValue value;
  FilterValue second_value;  // Used for BETWEEN condition
};

struct FilterNode {
  enum class Type { Logical, Condition, RawSQL } type;

  // For Logical nodes
  FilterOp op;
  std::vector<FilterNode> children;

  // For Condition nodes
  std::optional<FieldCondition> condition;

  // For RawSQL nodes
  std::optional<std::wstring> raw_sql;
};

class FilterSQLCompiler {
 public:
  struct Result {
    std::wstring where_clause;
    std::vector<FilterValue> params;
  };

  Result Compile(const FilterNode& node);

 private:
  Result CompileNode(const FilterNode& node);
  std::wstring FieldToColumn(FilterField field);
  std::wstring CompareToSQL(CompareOp op);
};

class FilterCombo {
 private:
  FilterNode _root;
  std::vector<SleeveFilter> _filters;

 public:
  filter_id_t filter_id;

};
};  // namespace puerhlab