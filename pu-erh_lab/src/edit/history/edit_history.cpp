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

#include "edit/history/edit_history.hpp"

#include <xxhash.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

#include "type/hash_type.hpp"
#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
VersionNode::VersionNode(Version& ver) : ver_ref_(ver) {}

/**
 * @brief Construct a new Edit History:: Edit History object
 *
 * @param bound_image
 */
EditHistory::EditHistory(sl_element_id_t bound_image) : bound_image_(bound_image) {
  SetAddTime();
  history_id_ = Hash128::Blend(Hash128::Compute(&added_time_, sizeof(added_time_)),
                               Hash128::Compute(&bound_image, sizeof(bound_image)));
}

void EditHistory::CalculateHistoryID() {
  auto& last_node = GetLatestVersion();
  // Merkle-like hash computation
  history_id_     = Hash128::Blend(
      history_id_,
      Hash128::Blend(last_node.ver_ref_.GetVersionID(),
                         Hash128::Compute(&last_modified_time_, sizeof(last_modified_time_))));
}
/**
 * @brief Set the created time for a EditHistory
 *
 */
void EditHistory::SetAddTime() {
  added_time_         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  last_modified_time_ = added_time_;
}

/**
 * @brief Update the last modified time stamp
 *
 */
void EditHistory::SetLastModifiedTime() {
  last_modified_time_ = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

auto EditHistory::GetAddTime() const -> std::time_t { return added_time_; }

auto EditHistory::GetLastModifiedTime() const -> std::time_t { return last_modified_time_; }

auto EditHistory::GetHistoryId() const -> history_id_t { return history_id_; }

auto EditHistory::GetBoundImage() const -> sl_element_id_t { return bound_image_; }

auto EditHistory::GetVersion(history_id_t ver_id) -> Version& {
  if (version_storage_.find(ver_id) == version_storage_.end()) {
    throw std::runtime_error("Version not found");
  }
  return version_storage_[ver_id];
}

auto EditHistory::CommitVersion(Version&& ver) -> history_id_t {
  auto ver_id = ver.GetVersionID();
  if (version_storage_.find(ver_id) != version_storage_.end()) {
    throw std::runtime_error("Version already exists");
  }
  version_storage_[ver_id] = std::move(ver);
  commit_tree_.emplace_back(version_storage_[ver_id]);
  commit_tree_.back().commit_id_ = static_cast<p_hash_t>(commit_tree_.size());
  SetLastModifiedTime();
  CalculateHistoryID();
  return ver_id;
}

auto EditHistory::GetLatestVersion() -> VersionNode& {
  if (commit_tree_.empty()) {
    throw std::runtime_error("No version in history");
  }
  return commit_tree_.back();
}

auto EditHistory::RemoveVersion(history_id_t ver_id) -> bool {
  if (version_storage_.find(ver_id) == version_storage_.end()) {
    return false;
  }
  // Remove from commit tree
  for (auto it = commit_tree_.begin(); it != commit_tree_.end(); ++it) {
    if (it->ver_ref_.GetVersionID() == ver_id) {  // TODO: Not support branching for now
      commit_tree_.erase(it);
      break;
    }
  }
  version_storage_.erase(ver_id);
  SetLastModifiedTime();
  return true;
}

auto EditHistory::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["history_id"]         = history_id_.ToString();
  j["bound_image"]        = bound_image_;
  j["added_time"]         = added_time_;
  j["last_modified_time"] = last_modified_time_;

  j["commit_tree"]        = nlohmann::json::array();
  for (const auto& node : commit_tree_) {
    nlohmann::json node_json;
    node_json["commit_id"] = node.commit_id_;
    node_json["version"]   = node.ver_ref_.ToJSON();
    j["commit_tree"].push_back(node_json);
  }

  j["version_storage"] = nlohmann::json::array();
  for (const auto& [ver_id, ver] : version_storage_) {
    nlohmann::json ver_json;
    ver_json["version_id"] = ver_id.ToString();
    ver_json["version"]    = ver.ToJSON();
    j["version_storage"].push_back(ver_json);
  }

  return j;
}

void EditHistory::FromJSON(const nlohmann::json& j) {
  if (!j.is_object() || !j.contains("history_id") || !j.contains("bound_image") ||
      !j.contains("added_time") || !j.contains("last_modified_time") ||
      !j.contains("commit_tree") || !j.contains("version_storage")) {
    throw std::runtime_error("EditHistory: Invalid JSON format for EditHistory");
  }

  history_id_         = Hash128::FromString(j.at("history_id").get<std::string>());
  bound_image_        = j.at("bound_image").get<sl_element_id_t>();
  added_time_         = j.at("added_time").get<std::time_t>();
  last_modified_time_ = j.at("last_modified_time").get<std::time_t>();
  commit_tree_.clear();
  version_storage_.clear();
  for (const auto& node_json : j.at("commit_tree")) {
    if (!node_json.is_object() || !node_json.contains("commit_id") ||
        !node_json.contains("version")) {
      commit_tree_.clear();
      version_storage_.clear();
      throw std::runtime_error(
          "EditHistory: Invalid JSON format for commit_tree node, clear all commit tree and "
          "version storage");
    }
    Version ver;
    ver.FromJSON(node_json.at("version"));
    history_id_t ver_id      = ver.GetVersionID();
    version_storage_[ver_id] = std::move(ver);
    VersionNode node(version_storage_[ver_id]);
    node.commit_id_ = node_json.at("commit_id").get<p_hash_t>();
    commit_tree_.push_back(std::move(node));
  }
}
};  // namespace puerhlab
