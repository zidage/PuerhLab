#pragma once

#include <duckdb.h>

#include <codecvt>
#include <filesystem>
#include <memory>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "storage/mapper/sleeve/element/element_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ElementService {
 private:
  duckdb_connection                                _conn;
  ElementMapper                                    _mapper;

  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  void InsertElementParams(const ElementMapperParams& param);

 public:
  explicit ElementService(duckdb_connection conn) : _conn(conn), _mapper(_conn) {}

  auto ToParams(const SleeveElement& source) -> ElementMapperParams;
  auto FromParams(const ElementMapperParams&& param) -> std::shared_ptr<SleeveElement>;

  void InsertElement(const SleeveElement& element);

  auto GetElementByPredicate(const std::wstring& predicate)
      -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementById(const sl_element_id_t id) -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementByName(const std::wstring name) -> std::vector<std::shared_ptr<SleeveElement>>;

  void RemoveElementById(sl_element_id_t id);

  void UpdateElement(const SleeveElement& updated);
};
};  // namespace puerhlab