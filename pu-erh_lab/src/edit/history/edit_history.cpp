#include "edit/history/edit_history.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include "type/hash_type.hpp"


#include <xxhash.h>

#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
VersionNode::VersionNode(Version& ver) : _ver_ref(ver) {}

/**
 * @brief Construct a new Edit History:: Edit History object
 *
 * @param bound_image
 */
EditHistory::EditHistory(sl_element_id_t bound_image) : _bound_image(bound_image) {
  SetAddTime();
  _history_id = Hash128(XXH3_128bits(this, sizeof(*this)));
}

/**
 * @brief Set the created time for a EditHistory
 *
 */
void EditHistory::SetAddTime() {
  _added_time         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  _last_modified_time = _added_time;
}

/**
 * @brief Update the last modified time stamp
 *
 */
void EditHistory::SetLastModifiedTime() {
  _last_modified_time = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

auto EditHistory::GetAddTime() const -> std::time_t { return _added_time; }

auto EditHistory::GetLastModifiedTime() const -> std::time_t { return _last_modified_time; }

auto EditHistory::GetHistoryId() const -> history_id_t { return _history_id; }

auto EditHistory::GetBoundImage() const -> sl_element_id_t { return _bound_image; }

auto EditHistory::GetVersion(history_id_t ver_id) -> Version& {
  if (_version_storage.find(ver_id) == _version_storage.end()) {
    throw std::runtime_error("Version not found");
  }
  return _version_storage[ver_id];
}

auto EditHistory::CommitVersion(Version&& ver) -> history_id_t {
  ver.CalculateVersionID();
  auto ver_id = ver.GetVersionID();
  if (_version_storage.find(ver_id) != _version_storage.end()) {
    throw std::runtime_error("Version already exists");
  }
  _version_storage[ver_id] = std::move(ver);
  _commit_tree.emplace_back(_version_storage[ver_id]);
  SetLastModifiedTime();
  return ver_id;
}

auto EditHistory::GetLatestVersion() -> VersionNode& {
  if (_commit_tree.empty()) {
    throw std::runtime_error("No version in history");
  }
  return _commit_tree.back();
}

auto EditHistory::RemoveVersion(history_id_t ver_id) -> bool {
  if (_version_storage.find(ver_id) == _version_storage.end()) {
    return false;
  }
  // Remove from commit tree
  for (auto it = _commit_tree.begin(); it != _commit_tree.end(); ++it) {
    if (it->_ver_ref.GetVersionID() == ver_id) {  // TODO: Not support branching for now
      _commit_tree.erase(it);
      break;
    }
  }
  _version_storage.erase(ver_id);
  SetLastModifiedTime();
  return true;
}

auto EditHistory::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["history_id"]         = _history_id.ToString();
  j["bound_image"]        = _bound_image;
  j["added_time"]         = _added_time;
  j["last_modified_time"] = _last_modified_time;

  j["commit_tree"]        = nlohmann::json::array();
  for (const auto& node : _commit_tree) {
    nlohmann::json node_json;
    node_json["commit_id"] = node._commit_id;
    node_json["version"]   = node._ver_ref.ToJSON();
    j["commit_tree"].push_back(node_json);
  }

  j["version_storage"] = nlohmann::json::array();
  for (const auto& [ver_id, ver] : _version_storage) {
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

  _history_id         = Hash128::FromString(j.at("history_id").get<std::string>());
  _bound_image        = j.at("bound_image").get<sl_element_id_t>();
  _added_time         = j.at("added_time").get<std::time_t>();
  _last_modified_time = j.at("last_modified_time").get<std::time_t>();
  _commit_tree.clear();
  _version_storage.clear();
  for (const auto& node_json : j.at("commit_tree")) {
    if (!node_json.is_object() || !node_json.contains("commit_id") ||
        !node_json.contains("version")) {
      _commit_tree.clear();
      _version_storage.clear();
      throw std::runtime_error(
          "EditHistory: Invalid JSON format for commit_tree node, clear all commit tree and "
          "version storage");
    }
    Version ver;
    ver.FromJSON(node_json.at("version"));
    history_id_t ver_id          = ver.GetVersionID();
    _version_storage[ver_id] = std::move(ver);
    VersionNode node(_version_storage[ver_id]);
    node._commit_id = node_json.at("commit_id").get<p_hash_t>();
    _commit_tree.push_back(std::move(node));
  }
}
};  // namespace puerhlab