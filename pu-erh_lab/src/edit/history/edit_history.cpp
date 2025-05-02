#include "edit/history/edit_history.hpp"

UUIDv4::UUIDGenerator<std::mt19937_64> uuid_generator;

namespace puerhlab {
EditHistory::EditHistory(image_id_t _bound_image) { _history_id = uuid_generator.getUUID(); }
};  // namespace puerhlab