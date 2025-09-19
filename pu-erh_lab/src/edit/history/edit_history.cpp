#include "edit/history/edit_history.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#include <xxhash.hpp>
#else
#include <xxhash.h>
#endif

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
#ifdef _WIN32
  _history_id = xxh::xxhash<64>(this, sizeof(*this));
#else
  _history_id = XXH64(this, sizeof(*this), 0);
#endif
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

auto EditHistory::GetHistoryId() const -> p_hash_t { return _history_id; }

auto EditHistory::GetBoundImage() const -> sl_element_id_t { return _bound_image; }

auto EditHistory::GetVersion(p_hash_t ver_id) -> Version& {
  if (_version_storage.find(ver_id) == _version_storage.end()) {
    throw std::runtime_error("Version not found");
  }
  return _version_storage[ver_id];
}

auto EditHistory::CommitVersion(Version&& ver) -> p_hash_t {
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

auto EditHistory::RemoveVersion(p_hash_t ver_id) -> bool {
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
};  // namespace puerhlab