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

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "filters/sleeve_filter.hpp"
#include "image/image.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class FilterOp { AND, OR, NOT };

/**
 * @brief Filterable fields in Image metadata
 *
 */
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

/**
 * @brief A minimal SQL compiler for FilterNode, used to generate WHERE clauses ONLY on Image table
 *
 */
class FilterSQLCompiler {
 public:
  struct Result {
    std::wstring             where_clause;
    std::vector<FilterValue> params;
  };

  static std::wstring Compile(const FilterNode& node);

 private:
  static inline std::wstring GenerateConditionString(const FieldCondition& cond);
  static std::wstring        CompileNode(const FilterNode& node);
  static std::wstring        FieldToColumn(FilterField field);
  static std::wstring        CompareToSQL(CompareOp op);
};

class FilterCombo {
 public:
  filter_id_t filter_id;

 private:
  FilterNode _root;

 public:
  FilterCombo() = default;
  FilterCombo(const filter_id_t id, const FilterNode& root) : filter_id(id), _root(root) {}

  const FilterNode& GetRoot() const { return _root; }

  void              SetRoot(const FilterNode& root) { _root = root; }

  auto              GenerateSQLOn(sl_element_id_t parent_id) const -> std::wstring;
};
};  // namespace puerhlab