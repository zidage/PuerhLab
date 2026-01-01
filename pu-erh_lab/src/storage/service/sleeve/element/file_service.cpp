//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "storage/service/sleeve/element/file_service.hpp"

#include <stdexcept>

namespace puerhlab {
auto FileService::ToParams(const std::pair<sl_element_id_t, image_id_t>& source)
    -> FileMapperParams {
  return {source.first, source.second};
}

auto FileService::FromParams(FileMapperParams&& param) -> std::pair<sl_element_id_t, image_id_t> {
  return {param.file_id_, param.image_id_};
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