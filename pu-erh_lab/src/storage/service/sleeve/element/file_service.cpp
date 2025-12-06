#include "storage/service/sleeve/element/file_service.hpp"

#include <stdexcept>

namespace puerhlab {
auto FileService::ToParams(const std::pair<sl_element_id_t, image_id_t>& source)
    -> FileMapperParams {
  return {source.first, source.second};
}

auto FileService::FromParams(FileMapperParams&& param)
    -> std::pair<sl_element_id_t, image_id_t> {
  return {param.file_id, param.image_id};
}

auto FileService::GetFileById(const sl_element_id_t id) -> std::pair<sl_element_id_t, image_id_t> {
  auto result = GetByPredicate(std::format("file_id={}", id).c_str());
  if (result.size() != 1) {
    throw std::runtime_error("File Service: Unable to recover a file image mapping: broken record");
  }
  return result.at(0);
}

auto FileService::GetBoundImageById(const sl_element_id_t id) -> image_id_t {
  return GetFileById(id).second;
}

void FileService::RemoveBindByFileId(const sl_element_id_t id) { RemoveById(id); }
};  // namespace puerhlab