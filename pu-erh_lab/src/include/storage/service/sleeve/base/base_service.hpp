#include <codecvt>
#include <memory>

#include "sleeve/sleeve_base.hpp"
#include "storage/mapper/sleeve/base/base_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {
class BaseService {
 private:
  duckdb_connection                                _conn;
  BaseMapper                                       _base_mapper;
  RootMapper                                       _root_mapper;

  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  void InsertElementParams(const BaseMapperParams& param);

 public:
  explicit BaseService(duckdb_connection conn)
      : _conn(conn), _base_mapper(_conn), _root_mapper(_conn) {}

  auto ToBaseParams(const SleeveBase& source) -> BaseMapperParams;
  auto FromBaseParams(const BaseMapperParams&& param) -> std::shared_ptr<SleeveBase>;
  auto ToRootParams(const SleeveBase& source) -> BaseMapperParams;
  auto FromRootParams(const RootMapperParams&& param) -> sl_element_id_t;

  void InsertElement(const SleeveElement& element);

  auto GetElementByPredicate(const std::wstring predicate)
      -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementById(const sl_element_id_t id) -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementByName(const std::wstring name) -> std::vector<std::shared_ptr<SleeveElement>>;

  void RemoveElementById(sl_element_id_t id);

  void UpdateElement(const SleeveElement& updated);
};
};  // namespace puerhlab