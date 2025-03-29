/*
 * @file        pu-erh_lab/src/include/mapper/sleeve/sleeve_mapper.hpp
 * @brief       Database interacting layer for sleeve base
 * @author      Yurun Zi
 * @date        2025-03-26
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "type/type.hpp"
#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_filter.hpp"
#include "image/image.hpp"

#include <duckdb.h>
#include <unordered_map>


namespace puerhlab {

/**
 * @brief Mapper interface for interacting with the DuckDB
 * 
 */
class SleeveMapper {
  void CreateDB(file_path_t db_path);

  void SetSleeve(SleeveBase sleeve_base);

  void AddImage(const Image &image);

  void UpdateImageById(image_id_t image_id, const Image &image);

  void RemoveImage(image_id_t image_id);

  void RemoveImageByFilter(const SleeveFilter &filter);

  auto GetSleeveBaseByFilter(const SleeveFilter &filter) -> std::unordered_map<image_id_t, Image>;  
};
};  // namespace puerhlab