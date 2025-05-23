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

#pragma once

#include <duckdb.h>

#include <memory>
#include <unordered_map>

#include "image/image.hpp"
#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {

class SleeveCaptureResources {
 private:
  void RecycleResources();

 public:
  duckdb_result             result;
  duckdb_prepared_statement stmt_base;
  duckdb_prepared_statement stmt_element;
  duckdb_prepared_statement stmt_root;
  duckdb_prepared_statement stmt_folder;
  duckdb_prepared_statement stmt_file;
  duckdb_prepared_statement stmt_filter;
  duckdb_prepared_statement stmt_history;
  duckdb_prepared_statement stmt_version;
  SleeveCaptureResources(duckdb_connection &con);

  ~SleeveCaptureResources();
};

/**
 * @brief Mapper interface for interacting with the DuckDB
 *
 */
class SleeveMapper {
 private:
  duckdb_database   _db;
  duckdb_connection _con;
  file_path_t       _db_path;
  bool              _db_connected = false;
  bool              _initialized  = false;

  sleeve_id_t       _captured_sleeve_id;
  bool              _has_sleeve = false;

  inline void       CaptureElement(std::shared_ptr<SleeveElement> element, SleeveCaptureResources &res);
  inline void       CaptureFolder(std::shared_ptr<SleeveFolder> folder, SleeveCaptureResources &res);
  inline void       CaptureFile(std::shared_ptr<SleeveFile> file, SleeveCaptureResources &res);
  inline void       CaptureFilters(std::unordered_map<uint32_t, std::shared_ptr<FilterCombo>> &filter_storage,
                                   SleeveCaptureResources                                     &res);

 public:
  explicit SleeveMapper();
  explicit SleeveMapper(file_path_t db_path);
  ~SleeveMapper();

  void ConnectDB(file_path_t db_path);
  void InitDB();
  void CaptureSleeve(const std::shared_ptr<SleeveBase> sleeve_base);
  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);
  void AddFilter(const std::shared_ptr<SleeveFolder> sleeve_folder, const std::shared_ptr<FilterCombo> filter);
  void AddImage(const std::shared_ptr<Image> image);
  void EditImage(const std::shared_ptr<Image> image, const image_id_t id);
  void RemoveImage(image_id_t image_id);
  void RestoreSleeveFromDB(sleeve_id_t sleeve_id);
  void RemoveSleeveBase(sleeve_id_t sleeve_id);
};
};  // namespace puerhlab