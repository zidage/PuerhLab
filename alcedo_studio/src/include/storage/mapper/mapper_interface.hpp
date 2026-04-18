//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <duckdb.h>

#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <vector>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"

namespace alcedo {
template <typename Derived, typename Mappable, typename ID>
class MapperInterface {
 public:
  duckdb_connection& conn_;

  MapperInterface(duckdb_connection& conn) : conn_(conn) {}

  /**
   * @brief Insert a new record into the table
   *
   * @param obj
   */
  void Insert(const Mappable&& obj) {
    duckorm::insert(conn_, Derived::TableName(), &obj, Derived::FieldDesc(), Derived::FieldCount());
  }

  /**
   * @brief Remove a record from the table by its primary key
   *
   * @param remove_id
   */
  void Remove(const ID remove_id) {
    std::string remove_clause = std::format(Derived::PrimeKeyClause(), remove_id);
    duckorm::remove(conn_, Derived::TableName(), remove_clause.c_str());
  }

  /**
   * @brief Remove records from the table by a custom SQL predicate
   *
   * @param predicate
   */
  void RemoveByClause(const std::string& predicate) {
    duckorm::remove(conn_, Derived::TableName(), predicate.c_str());
  }

  /**
   * @brief Get records from the table by a custom SQL predicate
   *
   * @param where_clause
   * @return std::vector<Mappable>
   */
  auto Get(const char* where_clause) -> std::vector<Mappable> {
    auto                  raw = duckorm::select(conn_, Derived::TableName(), Derived::FieldDesc(),
                                                Derived::FieldCount(), where_clause);
    std::vector<Mappable> result;
    for (auto& row : raw) {
      result.emplace_back(Derived::FromRawData(std::move(row)));
    }
    return result;
  }

  auto GetByQuery(const char* query) -> std::vector<Mappable> {
    auto raw = duckorm::select_by_query(conn_, Derived::FieldDesc(), Derived::FieldCount(), query);
    std::vector<Mappable> result;
    for (auto& row : raw) {
      result.emplace_back(Derived::FromRawData(std::move(row)));
    }
    return result;
  }

  /**
   * @brief Update a record in the table by its primary key
   *
   * @param target_id
   * @param updated
   */
  void Update(const ID target_id, const Mappable&& updated) {
    std::string where_clause = std::format(Derived::PrimeKeyClause(), target_id);
    duckorm::update(conn_, Derived::TableName(), &updated, Derived::FieldDesc(),
                    Derived::FieldCount(), where_clause.c_str());
  }
};

// Don't understand what heck this is... They call it CRTP (C++ Recurring Tremendous Pain, maybe).
template <typename Derived>
struct FieldReflectable {
 public:
  using FieldArrayType = std::span<const duckorm::DuckFieldDesc>;
  static constexpr FieldArrayType FieldDesc() { return Derived::field_descs_; }
  static constexpr uint32_t       FieldCount() { return Derived::field_count_; }
  static constexpr const char*    TableName() { return Derived::table_name_; }
  static constexpr const char*    PrimeKeyClause() { return Derived::prime_key_clause_; }
};
};  // namespace alcedo