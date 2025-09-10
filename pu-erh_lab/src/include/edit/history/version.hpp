/*
 * @file        pu-erh_lab/src/include/edit/history/version.hpp
 * @brief       A snapshot of a edit version
 * @author      Yurun Zi
 * @date        2025-03-23
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

#include <ctime>
#include <list>
#include <vector>

#include "edit/history/edit_transaction.hpp"
#include "edit_transaction.hpp"
#include "type/type.hpp"

namespace puerhlab {
class Version {
 private:
  /**
   * @brief MurmurHash3 value for this version
   */
  p_hash_t                   _version_id;
  /**
   * @brief Last modified time for this version
   */
  std::time_t                _added_time;
  std::time_t                _last_modified_time;
  /**
   * @brief collection of images related to this version
   */
  sl_element_id_t            _bound_image;
  /**
   * @brief Edit transactions for this edit version
   */
  std::list<EditTransaction> _edit_transactions;

 public:
  Version(sl_element_id_t bound_image);

  void CalculateVersionID();
  auto GetVersionID() const -> p_hash_t;

  void SetAddTime();
  auto GetAddTime() const -> std::time_t;
  void SetLastModifiedTime();
  auto GetLastModifiedTime() const -> std::time_t;

  void SetBoundImage(sl_element_id_t bound_image);
  auto GetBoundImage() const -> sl_element_id_t;

  void AppendEditTransaction(EditTransaction&& edit_transaction);
  auto RemoveLastEditTransaction() -> EditTransaction;
  auto GetLastEditTransaction() -> EditTransaction&;
  auto GetAllEditTransactions() const -> const std::list<EditTransaction>&;
};
};  // namespace puerhlab