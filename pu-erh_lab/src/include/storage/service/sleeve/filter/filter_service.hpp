#pragma once

#include <memory>
#include <vector>

#include "sleeve/sleeve_filter/filters/sleeve_filter.hpp"
#include "storage/mapper/sleeve/filter/filter_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {

class FilterService {
  duckdb_connection _conn;

  FilterMapper      _filter_mapper;

  void              InsertFilterParams(const FilterMapperParams& param);

 public:
  explicit FilterService(duckdb_connection conn) : _conn(conn), _filter_mapper(_conn) {}

  auto ToParams(const SleeveFilter& source) -> FilterMapperParams;
  auto FromParams(const FilterMapperParams&& param) -> std::shared_ptr<SleeveFilter>;

  void InsertFilter(const SleeveFilter& filter);

  auto GetFilterById(const filter_id_t id) -> std::shared_ptr<SleeveFilter>;
  auto GetFilterByType(const FilterType type) -> std::vector<std::shared_ptr<SleeveFilter>>;

  void RemoveFilterById(const filter_id_t id);

  void UpdateFilter(const SleeveFilter& updated);
};
};  // namespace puerhlab