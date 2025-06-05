#include "storage/mapper/sleeve/file_mapper.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"

namespace puerhlab {
auto FileMapper::FromDesc(std::vector<DuckFieldDesc>&& fields) -> FileMapperParams {
  if (fields.size() != 2) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveFile");
  }
  std::shared_ptr<SleeveFile> file = std::make_shared<SleeveFile>();
}
};  // namespace puerhlab