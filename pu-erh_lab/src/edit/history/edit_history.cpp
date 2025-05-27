#include "edit/history/edit_history.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <xxhash.hpp>

#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
/**
 * @brief Construct a new Edit History:: Edit History object
 *
 * @param bound_image
 */
EditHistory::EditHistory(sl_element_id_t bound_image) : _bound_image(bound_image) {
  SetAddTime();
  _history_id = xxh::xxhash<64>(this, sizeof(*this));
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
};  // namespace puerhlab