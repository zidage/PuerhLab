#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/mapper/sleeve/element/folder_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {

class FolderService
    : public ServiceInterface<FolderService, std::pair<sl_element_id_t, sl_element_id_t>,
                              FolderMapperParams, FolderMapper, sl_element_id_t> {
  using ServiceInterface::ServiceInterface;

 public:
  // File service is used to retrieve a set of mapping, therefore no actual internal type, i.g.
  // SleeveFile, is returned
  static auto ToParams(const std::pair<sl_element_id_t, sl_element_id_t> source)
      -> FolderMapperParams;
  static auto FromParams(const FolderMapperParams&& param)
      -> std::pair<sl_element_id_t, sl_element_id_t>;

  auto GetFolderContent(const sl_element_id_t id) -> std::vector<sl_element_id_t>;
  auto GetFolderByContentId(const sl_element_id_t id) -> sl_element_id_t;

  void RemoveAllContents(const sl_element_id_t folder_id);
  void RemoveContentById(const sl_element_id_t content_id);

  void UpdateFolderContent(const std::vector<sl_element_id_t>& content,
                           const sl_element_id_t               folder_id);
};
};  // namespace puerhlab