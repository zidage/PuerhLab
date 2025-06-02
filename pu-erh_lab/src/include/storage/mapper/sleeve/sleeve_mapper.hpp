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

#include <codecvt>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "image/image.hpp"
#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/sleeve/statement_prepare.hpp"
#include "type/type.hpp"

#define GET_OP(x, y) (static_cast<uint8_t>(x) | static_cast<uint8_t>(y))

namespace puerhlab {
/**
 * @brief A table operation is represented by five binary digits.
 * The most-significant 2 digits represents the operation type, and
 * the least-significant 3 digits represents the target table.
 * Example: To add an image to the image table, the operation
 * ADD_IMAGE is represented by binary number 00000 == Operate | Table.
 * TO edit an filter, the operation EDIT_FILTER is
 * 10101 = EDIT | FILTER
 */
enum class Table : uint8_t {
  Image       = 0x0,
  Element     = 0x1,
  File        = 0x2,
  Folder      = 0x3,
  ComboFolder = 0x4,
  Filter      = 0x5,
  History     = 0x6,
  Version     = 0x7
};

enum class Operate : uint8_t { ADD = 0x0, DELETE = 0x8, EDIT = 0x10, LOOKUP = 0x18 };
/**
 * @brief Mapper interface for interacting with the DuckDB
 *
 */
class SleeveMapper {
 private:
  duckdb_database                                  _db;
  duckdb_connection                                _con;
  file_path_t                                      _db_path;
  bool                                             _db_connected = false;
  bool                                             _initialized  = false;

  sleeve_id_t                                      _captured_sleeve_id;
  bool                                             _has_sleeve = false;

  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

  std::vector<Prepare>                             _prepare_storage;

  inline void CaptureElement(std::unordered_map<uint32_t, std::shared_ptr<SleeveElement>> &storage, Prepare &pre);
  inline void CaptureFolder(std::shared_ptr<SleeveFolder> folder, Prepare &pre);
  inline void CaptureFile(std::shared_ptr<SleeveFile> file, Prepare &pre);
  inline void CaptureFilters(std::unordered_map<uint32_t, std::shared_ptr<FilterCombo>> &filter_storage, Prepare &pre);

 public:
  explicit SleeveMapper();
  explicit SleeveMapper(file_path_t db_path);
  ~SleeveMapper();

  void ConnectDB(file_path_t db_path);
  void InitDB();

  auto GetPrepare(uint8_t op, const std::string &query) -> Prepare &;

  void CaptureSleeve(const std::shared_ptr<SleeveBase> sleeve_base, const std::shared_ptr<ImagePoolManager> image_pool);
  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);

  void AddImage(const Image &image);
  auto GetImage(const image_id_t id) -> std::shared_ptr<Image>;
  void EditImage(const Image &image, const image_id_t id);
  void RemoveImage(const image_id_t image_id);

  void AddElement(const std::shared_ptr<SleeveElement> element);
  void GetElement(const sl_element_id_t element_id);
  void RemoveElement(const sl_element_id_t element_id);
  void EditElement(const sl_element_id_t element_id, const std::shared_ptr<SleeveElement> element);

  void AddFolder(const std::shared_ptr<SleeveFolder> folder);
  void AddFile(const std::shared_ptr<SleeveFile> file);

  void RemoveFolder(const sl_element_id_t folder_id);
  void RemoveFile(const sl_element_id_t file_id);

  void AddFilter(const sl_element_id_t folder_id, const std::shared_ptr<FilterCombo> filterd);
  void RemoveFilterCombo(const sl_element_id_t combo_id);

  void RestoreSleeveFromDB(sleeve_id_t sleeve_id);
  void RemoveSleeveBase(sleeve_id_t sleeve_id);
};
};  // namespace puerhlab