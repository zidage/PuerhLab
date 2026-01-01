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

#include "storage/service/sleeve/element/folder_service.hpp"

#include <format>
#include <unordered_map>
#include <vector>

#include "type/type.hpp"

namespace puerhlab {
auto FolderService::ToParams(const std::pair<sl_element_id_t, sl_element_id_t> source)
    -> FolderMapperParams {
  return {source.first, source.second};
}

auto FolderService::FromParams(FolderMapperParams&& param)
    -> std::pair<sl_element_id_t, sl_element_id_t> {
  return {param.folder_id_, param.element_id_};
}

auto FolderService::GetFolderContent(const sl_element_id_t id) -> std::vector<sl_element_id_t> {
  auto                         results = GetByPredicate(std::format("folder_id={}", id));
  std::vector<sl_element_id_t> folder_content;
  folder_content.resize(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    folder_content[i] = results[i].second;
  }
  return folder_content;
}

void FolderService::RemoveAllContents(const sl_element_id_t folder_id) { RemoveById(folder_id); }

void FolderService::RemoveContentById(const sl_element_id_t content_id) {
  RemoveByClause(std::format("element_id={}", content_id));
}

void FolderService::UpdateFolderContent(const std::vector<sl_element_id_t>& content,
                                        const sl_element_id_t               folder_id) {
  RemoveAllContents(folder_id);
  for (auto& element_id : content) {
    Insert({folder_id, element_id});
  }
}
};  // namespace puerhlab