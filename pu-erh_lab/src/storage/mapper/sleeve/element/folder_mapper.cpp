#include "storage/mapper/sleeve/element/folder_mapper.hpp"

#include <format>
#include <stdexcept>
#include <variant>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FolderMapper::FromRawData(std::vector<VarTypes>&& data) -> FolderMapperParams {
  if (data.size() != 2) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveFolder");
  }

  auto folder_id    = std::get_if<sl_element_id_t>(&data[0]);
  auto element_name = std::get_if<const char*>(&data[1]);
  auto element_id   = std::get_if<sl_element_id_t>(&data[2]);

  if (folder_id == nullptr || element_name == nullptr || element_id == nullptr) {
    throw std::runtime_error("Unmatching types occured when parsing the data from the DB");
  }

  return {*folder_id, *element_name, *element_id};
}

void FolderMapper::Insert(const FolderMapperParams params) {
  //   SELECT * FROM FolderContent WHERE folder_id = ?;
  duckorm::insert(_conn, "FolderContent", &params, GetDesc(), 3);
}

auto FolderMapper::Get(const sl_element_id_t id) -> std::vector<FolderMapperParams> {
  // SELECT * FROM Image WHERE id = ?;
  std::vector<std::vector<VarTypes>> raw_results = duckorm::select<std::vector<VarTypes>>(
      _conn, "FolderContent", GetDesc(), 3, std::format("folder_id={}", id).c_str());
  std::vector<FolderMapperParams> result;
  for (auto& raw : raw_results) {
    result.emplace_back(FromRawData(std::move(raw)));
  }

  return result;
}

auto FolderMapper::Get(const char* where_clause) -> std::vector<FolderMapperParams> {
  // SELECT * FROM Image WHERE id = ?;
  std::vector<std::vector<VarTypes>> raw_results =
      duckorm::select<std::vector<VarTypes>>(_conn, "FolderContent", GetDesc(), 3, where_clause);
  std::vector<FolderMapperParams> result;
  for (auto& raw : raw_results) {
    result.emplace_back(FromRawData(std::move(raw)));
  }
  return result;
}

void FolderMapper::Remove(const sl_element_id_t id) {
  // DELETE FROM FolderContent WHERE folder_id = ?;
  duckorm::remove(_conn, "FolderContent", std::format("folder_id={}", id).c_str());
}

/**
 * @brief MUST CALL Remove() BEFORE ANY UPDATE!
          It is recommended to use Remove() and Insert() to replace Update() when using FolderMapper
 *
 * @param element_id
 * @param updated
 */
void FolderMapper::Update(const sl_element_id_t id, const FolderMapperParams updated) {
  duckorm::update(_conn, "FileImage", &updated, GetDesc(), 3,
                  std::format("folder_id={}", id).c_str());
}
};  // namespace puerhlab