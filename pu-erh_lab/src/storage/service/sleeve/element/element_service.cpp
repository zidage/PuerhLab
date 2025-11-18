#include "storage/service/sleeve/element/element_service.hpp"

#include <cstdint>
#include <ctime>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
auto ElementService::ToParams(const std::shared_ptr<SleeveElement>& source) -> ElementMapperParams {
  char added_time[32];
  char modified_time[32];
  std::strftime(added_time, sizeof(added_time), "%Y-%m-%d %H:%M:%S",
                std::gmtime(&source->_added_time));
  std::strftime(modified_time, sizeof(modified_time), "%Y-%m-%d %H:%M:%S",
                std::gmtime(&source->_last_modified_time));

  std::string utf_8_str = conv::ToBytes(source->_element_name);
  return {source->_element_id,
          static_cast<uint32_t>(source->_type),
          std::make_unique<std::string>(utf_8_str),
          std::make_unique<std::string>(added_time),
          std::make_unique<std::string>(modified_time),
          source->_ref_count};
}

auto ElementService::FromParams(const ElementMapperParams&& param)
    -> std::shared_ptr<SleeveElement> {
  auto               id                = param.id;
  auto               type              = static_cast<ElementType>(param.type);
  auto               element_name      = conv::FromBytes(*param.element_name);
  auto               added_time_str    = *param.added_time;
  auto               modified_time_str = *param.modified_time;
  auto               ref_count         = param.ref_count;

  std::tm            tm_added{};
  std::tm            tm_modified{};

  std::istringstream a_ss(added_time_str);
  std::istringstream m_ss(modified_time_str);
  a_ss >> std::get_time(&tm_added, "%Y-%m-%d %H:%M:%S");
  m_ss >> std::get_time(&tm_modified, "%Y-%m-%d %H:%M:%S");
  std::shared_ptr<SleeveElement> element;
  if (type == ElementType::FILE) {
    element = std::make_shared<SleeveFile>(id, element_name);
  } else if (type == ElementType::FOLDER) {
    element = std::make_shared<SleeveFolder>(id, element_name);
  } else {
    throw std::runtime_error("ElementService: Invalid ElementMapperParams");
  }

  element->_added_time         = std::mktime(&tm_added);
  element->_last_modified_time = std::mktime(&tm_modified);
  element->_ref_count          = ref_count;

  return element;
}

auto ElementService::GetElementById(const sl_element_id_t id) -> std::shared_ptr<SleeveElement> {
  auto result = GetByPredicate(std::format("id={}", id));
  if (result.size() > 1) {
    throw std::runtime_error("Element Service: Sleeve element id is not unique. Broken DB file");
  }
  if (result.size() == 0) {
    throw std::runtime_error(
        std::format("Element Service: No element with id {} is stored in DB", id));
  }
  return result[0];
}

auto ElementService::GetElementByName(const std::wstring name)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  std::string predicate = conv::ToBytes(std::format(L"element_name={}", name));
  return GetByPredicate(std::move(predicate));
}

auto ElementService::GetElementByType(const ElementType type)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  std::string predicate = std::format("type={}", static_cast<uint32_t>(type));
  return GetByPredicate(std::move(predicate));
}

auto ElementService::GetElementsInFolderByFilter(const std::wstring filter_sql)
    -> std::vector<std::shared_ptr<SleeveElement>> {
  std::string filter_sql_u8 = conv::ToBytes(filter_sql);
  return GetByQuery(std::move(filter_sql_u8));
}
};  // namespace puerhlab