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

#include <cstddef>
#include <memory>
#include <vector>

#include "storage/mapper/mapper_interface.hpp"

namespace puerhlab {
template <typename Derived, typename InternalType, typename Mappable, typename Mapper, typename ID>
class ServiceInterface {
 private:
  duckdb_connection&                    conn_;
  MapperInterface<Mapper, Mappable, ID> mapper_;

 public:
  ServiceInterface(duckdb_connection& conn) : conn_(conn), mapper_(conn) {}
  void InsertParams(const Mappable& param) { mapper_.Insert(std::move(param)); }
  void Insert(const InternalType& obj) { mapper_.Insert(Derived::ToParams(obj)); }

  /**
   * @brief Get the objects by a SQL predicate (WHERE clause)
   *
   * @param predicate
   * @return std::vector<InternalType>
   */
  auto GetByPredicate(std::string&& predicate) -> std::vector<InternalType> {
    std::vector<Mappable>     param_results = mapper_.Get(predicate.c_str());
    std::vector<InternalType> results;
    results.resize(param_results.size());
    size_t idx = 0;
    for (auto& param : param_results) {
      results[idx] = Derived::FromParams(std::move(param));
      ++idx;
    }
    return results;
  }

  /**
   * @brief Get the objects by a full SQL query, results may not compatible with Mappable.
   *        Should only be used internally (e.g., filter).
   *
   * @param query
   * @return std::vector<InternalType>
   */
  auto GetByQuery(std::string&& query) -> std::vector<InternalType> {
    std::vector<Mappable>     param_results = mapper_.GetByQuery(query.c_str());
    std::vector<InternalType> results;
    results.resize(param_results.size());
    size_t idx = 0;
    for (auto& param : param_results) {
      results[idx] = Derived::FromParams(std::move(param));
      ++idx;
    }
    return results;
  }

  void RemoveById(const ID remove_id) { mapper_.Remove(remove_id); }
  void RemoveByClause(const std::string& clause) { mapper_.RemoveByClause(clause); }
  void Update(const InternalType& obj, const ID update_id) {
    mapper_.Update(update_id, Derived::ToParams(obj));
  }
};
}  // namespace puerhlab