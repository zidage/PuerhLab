/*
 * @file        pu-erh_lab/src/include/edit/history/edit_history.hpp
 * @brief       Data structure used to track all the edit history
 * @author      Yurun Zi
 * @date        2025-03-23
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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