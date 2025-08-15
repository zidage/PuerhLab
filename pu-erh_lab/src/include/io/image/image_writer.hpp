#pragma once

#include <filesystem>
#include <vector>

#include "sleeve/sleeve_filesystem.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImageWriter {
 private:
  image_path_t                 _output_path;

  std::vector<sl_element_id_t> _output_file_ids;

  FileSystem&                  _file_system;

 public:
  ImageWriter() = delete;
  ImageWriter(const image_path_t& output_path, FileSystem& file_system);
};
};  // namespace puerhlab