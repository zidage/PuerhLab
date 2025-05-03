#include "edit/history/edit_history.hpp"

#include <chrono>

#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"

static UUIDv4::UUIDGenerator<std::mt19937_64> uuid_generator;

namespace puerhlab {
/**
 * @brief Construct a new Edit History:: Edit History object
 *
 * @param bound_image
 */
EditHistory::EditHistory(sl_element_id_t bound_image) : _bound_image(bound_image) {
  _history_id = uuid_generator.getUUID();
  SetAddTime();
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