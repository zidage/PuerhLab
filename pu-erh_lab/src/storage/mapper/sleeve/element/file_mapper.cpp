#include "storage/mapper/sleeve/element/file_mapper.hpp"

#include <duckdb.h>

#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FileMapper::FromRawData(std::vector<VarTypes>&& data) -> FileMapperParams {
  if (data.size() != 2) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveFile");
  }

  auto file_id = std::get_if<sl_element_id_t>(&data[0]);
  auto img_id  = std::get_if<image_id_t>(&data[1]);
  if (file_id == nullptr || img_id == nullptr) {
    throw std::runtime_error("Unmatching types occured when parsing the data from the DB");
  }
  return {*file_id, *img_id};
}

void FileMapper::Insert(const FileMapperParams params) {
  // INSERT INTO FileImage (file_id,image_id) VALUES (?, ?)"
  duckorm::insert(_conn, "FileImage", &params, GetDesc(), 2);
}

auto FileMapper::Get(const sl_element_id_t id) -> std::vector<FileMapperParams> {
  // SELECT * FROM Image WHERE id = ?;
  std::vector<std::vector<VarTypes>> raw_results = duckorm::select<std::vector<VarTypes>>(
      _conn, "FileImage", GetDesc(), 2, std::format("id={}", id).c_str());
  std::vector<FileMapperParams> result;
  for (auto& raw : raw_results) {
    result.emplace_back(FromRawData(std::move(raw)));
  }

  return result;
}

auto FileMapper::Get(const char* where_clause) -> std::vector<FileMapperParams> {
  // SELECT * FROM Image WHERE id = ?;
  std::vector<std::vector<VarTypes>> raw_results =
      duckorm::select<std::vector<VarTypes>>(_conn, "FileImage", GetDesc(), 2, where_clause);
  std::vector<FileMapperParams> result;
  for (auto& raw : raw_results) {
    result.emplace_back(FromRawData(std::move(raw)));
  }

  return result;
}

void FileMapper::Remove(const sl_element_id_t id) {
  // DELETE FROM Image WHERE id = ?;
  duckorm::remove(_conn, "FileImage", std::format("id={}", id).c_str());
}

void FileMapper::Update(const sl_element_id_t element_id, const FileMapperParams updated) {
  // UPDATE Image SET image_path = ?, file_name = ?, type = ?, metadata = ? WHERE id = ?;
  duckorm::update(_conn, "FileImage", &updated, GetDesc(), 2,
                  std::format("id={}", element_id).c_str());
}
};  // namespace puerhlab