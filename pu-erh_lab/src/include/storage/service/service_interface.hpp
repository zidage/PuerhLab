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
  duckdb_connection&                    _conn;
  MapperInterface<Mapper, Mappable, ID> _mapper;

 public:
  ServiceInterface(duckdb_connection& conn) : _conn(conn), _mapper(conn) {}
  void InsertParams(const Mappable& param) { _mapper.Insert(std::move(param)); }
  void Insert(const InternalType& obj) { _mapper.Insert(Derived::ToParams(obj)); }

  /**
   * @brief Get the objects by a SQL predicate (WHERE clause)
   *
   * @param predicate
   * @return std::vector<InternalType>
   */
  auto GetByPredicate(std::string&& predicate) -> std::vector<InternalType> {
    std::vector<Mappable>     param_results = _mapper.Get(predicate.c_str());
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
    std::vector<Mappable>     param_results = _mapper.GetByQuery(query.c_str());
    std::vector<InternalType> results;
    results.resize(param_results.size());
    size_t idx = 0;
    for (auto& param : param_results) {
      results[idx] = Derived::FromParams(std::move(param));
      ++idx;
    }
    return results;
  }

  void RemoveById(const ID remove_id) { _mapper.Remove(remove_id); }
  void RemoveByClause(const std::string& clause) { _mapper.RemoveByClause(clause); }
  void Update(const InternalType& obj, const ID update_id) {
    _mapper.Update(update_id, Derived::ToParams(obj));
  }
};
}  // namespace puerhlab