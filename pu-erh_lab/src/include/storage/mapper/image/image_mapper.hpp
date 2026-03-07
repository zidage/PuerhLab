//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstdint>
#include <memory>

#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE Image (id BIGINT PRIMARY KEY, image_path TEXT, file_name TEXT, type INTEGER,
// metadata JSON);
struct ImageMapperParams {
  image_id_t                   id;
  std::unique_ptr<std::string> image_path;
  std::unique_ptr<std::string> file_name;
  uint32_t                     type;
  std::unique_ptr<std::string> metadata;
};

class ImageMapper : public MapperInterface<ImageMapper, ImageMapperParams, image_id_t>,
                    public FieldReflectable<ImageMapper> {
 private:
  static constexpr uint32_t                                         field_count_      = 5;
  static constexpr const char*                                      table_name_       = "Image";
  static constexpr const char*                                      prime_key_clause_ = "id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, field_count_> field_descs_      = {
      FIELD(ImageMapperParams, id, UINT32), FIELD(ImageMapperParams, image_path, VARCHAR),
      FIELD(ImageMapperParams, file_name, VARCHAR), FIELD(ImageMapperParams, type, UINT32),
      FIELD(ImageMapperParams, metadata, VARCHAR)};

 public:
  static auto FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageMapperParams;
  friend struct FieldReflectable<ImageMapper>;
  using MapperInterface::MapperInterface;
};
}  // namespace puerhlab