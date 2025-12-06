#pragma once

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/sleeve/element/file_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
class FileService : public ServiceInterface<FileService, std::pair<sl_element_id_t, image_id_t>,
                                            FileMapperParams, FileMapper, sl_element_id_t> {
 public:
  using ServiceInterface::ServiceInterface;

  // File service is used to retrieve a set of mapping, therefore no actual internal type, i.g.
  // SleeveFile, is returned
  static auto ToParams(const std::pair<sl_element_id_t, image_id_t>& source) -> FileMapperParams;
  static auto FromParams(FileMapperParams&& param) -> std::pair<sl_element_id_t, image_id_t>;

  auto        GetFileById(const sl_element_id_t id) -> std::pair<sl_element_id_t, image_id_t>;
  auto        GetBoundImageById(const sl_element_id_t id) -> image_id_t;

  void        RemoveBindByFileId(const sl_element_id_t id);
};
};  // namespace puerhlab