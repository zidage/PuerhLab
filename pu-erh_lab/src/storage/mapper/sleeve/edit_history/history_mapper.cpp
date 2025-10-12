#include "storage/mapper/sleeve/edit_history/history_mapper.hpp"

namespace puerhlab {
auto EditHistoryMapper::FromRawData(std::vector<duckorm::VarTypes>&& data)
    -> EditHistoryMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for EditHistory");
  }
  auto file_id = std::get_if<sl_element_id_t>(&data[0]);
  auto history = std::get_if<std::unique_ptr<std::string>>(&data[1]);

  if (file_id == nullptr || history == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*file_id, std::move(*history)};
}
};  // namespace puerhlab