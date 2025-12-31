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

#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>

#include "edit/pipeline/pipeline.hpp"
#include "type/hash_type.hpp"
#include "type/type.hpp"
#include "version.hpp"

#pragma once

namespace puerhlab {
class VersionNode {
 public:
  Version& _ver_ref;
  // std::list<VersionNode> _branch; // TODO: Not support branching for now
  p_hash_t _commit_id;

 public:
  VersionNode(Version& ver_ref);
};

using history_id_t = Hash128;
/**
 * @brief A history of edits applied to a specific image. Each EditHistory instance is
 *        associated with a single image and tracks all changes made to it over time.
 *        It maintains a collection of Version instances, each representing a snapshot of the
 *        image at a specific point in time. This class is serializable to JSON.
 */
class EditHistory {
 private:
  history_id_t                              _history_id;
  sl_element_id_t                           _bound_image;

  std::time_t                               _added_time;
  std::time_t                               _last_modified_time;

  std::list<VersionNode>                    _commit_tree;

  std::unordered_map<history_id_t, Version> _version_storage;

  void                                      CalculateHistoryID();

 public:
  EditHistory(sl_element_id_t bound_image);
  void SetAddTime();
  void SetLastModifiedTime();
  auto GetAddTime() const -> std::time_t;
  auto GetLastModifiedTime() const -> std::time_t;

  auto GetHistoryId() const -> history_id_t;
  auto GetBoundImage() const -> sl_element_id_t;

  auto GetVersion(history_id_t ver_id) -> Version&;
  auto CommitVersion(Version&& ver) -> history_id_t;

  auto GetLatestVersion() -> VersionNode&;
  auto RemoveVersion(history_id_t ver_id) -> bool;

  auto ToJSON() const -> nlohmann::json;
  void FromJSON(const nlohmann::json& j);
};
};  // namespace puerhlab