#pragma once

#include <array>

#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE Sleeve (id BIGINT PRIMARY KEY);
struct BaseMapperParams {
  sleeve_id_t id;
};
class BaseMapper : MapperInterface<BaseMapperParams, sl_element_id_t>,
                   FieldReflectable<BaseMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> BaseMapperParams;

  static constexpr std::array<duckorm::DuckFieldDesc, 1> kFieldDescs = {
      FIELD(BaseMapperParams, id, UINT32)};

 public:
  friend struct FieldReflectable<BaseMapper>;
  using MapperInterface<BaseMapperParams, sl_element_id_t>::MapperInterface;

  void Insert(const BaseMapperParams&& params);
  auto Get(const sleeve_id_t id) -> std::vector<BaseMapperParams>;
  auto Get(const char* where_clause) -> std::vector<BaseMapperParams>;
  void Remove(const sleeve_id_t id);
  void Update(const sleeve_id_t id, const BaseMapperParams updated);
};

struct RootMapperParams {
  sl_element_id_t id;
};
class RootMapper : MapperInterface<RootMapperParams, sl_element_id_t>,
                   FieldReflectable<RootMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> RootMapperParams;

  static constexpr std::array<duckorm::DuckFieldDesc, 1> kFieldDescs = {
      FIELD(RootMapperParams, id, UINT32)};

 public:
  friend struct FieldReflectable<BaseMapper>;
  using MapperInterface<RootMapperParams, sl_element_id_t>::MapperInterface;

  void Insert(const RootMapperParams&& params);
  auto Get(const sl_element_id_t id) -> std::vector<RootMapperParams>;
  auto Get(const char* where_clause) -> std::vector<RootMapperParams>;
  void Remove(const sl_element_id_t id);
  void Update(const sl_element_id_t id, const RootMapperParams updated);
};
};  // namespace puerhlab