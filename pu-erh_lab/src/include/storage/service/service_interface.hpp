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
  _duckdb_connection                    _conn;
  MapperInterface<Mapper, Mappable, ID> _mapper;

  void InsertParams(const Mappable& param) { _mapper.Insert(std::move(param)); }

 public:
  void Insert(const InternalType& obj) { _mapper.Insert(Derived::ToParams(obj)); }
  auto Get(const std::string& predicate) -> std::vector<std::shared_ptr<InternalType>> {
    std::vector<Mapper>                        param_results = _mapper.Get(predicate);
    std::vector<std::shared_ptr<InternalType>> results;
    results.resize(param_results.size());
    size_t idx = 0;
    for (auto& param : param_results) {
      results[idx] = Derived::FromParams(std::move(param));
      ++idx;
    }
  }
};
}  // namespace puerhlab