#pragma once

#include <memory>
#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve/sleeve_filter/filters/sleeve_filter.hpp"
#include "storage/mapper/sleeve/filter/filter_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {

class ComboService {
  duckdb_connection _conn;

  ComboMapper       _mapper;

  void              InsertComboParams(const ComboMapperParams& param);

 public:
  explicit ComboService(duckdb_connection conn) : _conn(conn), _mapper(_conn) {}

  auto ToParams(const FilterCombo& source) -> ComboMapperParams;
  auto FromParams(const ComboMapperParams&& param) -> std::shared_ptr<FilterCombo>;

  void InsertCombo(const FilterCombo& filter);

  auto GetComboById(const filter_id_t id) -> std::shared_ptr<FilterCombo>;

  void RemoveComboById(const filter_id_t id);

  void UpdateCombo(const FilterCombo& updated);
};
};  // namespace puerhlab