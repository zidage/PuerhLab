#pragma once

#include <duckdb.h>

#include <codecvt>
#include <filesystem>
#include <memory>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "storage/mapper/sleeve/element/element_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ElementService
    : public ServiceInterface<ElementService, std::shared_ptr<SleeveElement>, ElementMapperParams,
                              ElementMapper, sl_element_id_t> {
 private:
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

 public:
  using ServiceInterface::ServiceInterface;
  auto ToParams(const SleeveElement& source) -> ElementMapperParams;
  auto FromParams(const ElementMapperParams&& param) -> std::shared_ptr<SleeveElement>;

  auto GetElementByName(const std::wstring name) -> std::vector<std::shared_ptr<SleeveElement>>;
  auto GetElementByType(const ElementType type) -> std::vector<std::shared_ptr<SleeveElement>>;
};
};  // namespace puerhlab