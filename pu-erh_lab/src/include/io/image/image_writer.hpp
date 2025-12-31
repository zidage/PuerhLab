//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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