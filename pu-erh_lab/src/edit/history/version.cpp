/*
 * @file        pu-erh_lab/src/edit/history/version.cpp
 * @brief       define behaviors of a version object
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

#include "edit/history/version.hpp"

#include <xxhash.hpp>

#include "utils/clock/time_provider.hpp"

namespace puerhlab {
Version::Version(sl_element_id_t bound_image) : _bound_image(bound_image) {
  _added_time         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  _last_modified_time = _added_time;
  _version_id         = 0;  // TODO: Generate hash value
}

void Version::CalculateVersionID() {
  SetLastModifiedTime();
  _version_id = xxh::xxhash<64>(this, sizeof(*this));
}

auto Version::GetVersionID() const -> p_hash_t { return _version_id; }

auto Version::GetAddTime() const -> std::time_t { return _added_time; }

auto Version::GetLastModifiedTime() const -> std::time_t { return _last_modified_time; }

void Version::SetLastModifiedTime() {
  _last_modified_time = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

void Version::SetBoundImage(sl_element_id_t image_id) { _bound_image = image_id; }

auto Version::GetBoundImage() const -> sl_element_id_t { return _bound_image; }

void Version::AppendEditTransaction(EditTransaction&& edit_transaction) {
  _edit_transactions.push_back(std::move(edit_transaction));
  SetLastModifiedTime();
}

auto Version::RemoveLastEditTransaction() -> EditTransaction {
  if (_edit_transactions.empty()) {
    throw std::runtime_error("Version: No edit transaction to remove");
  }
  EditTransaction last_transaction = std::move(_edit_transactions.back());
  _edit_transactions.pop_back();
  SetLastModifiedTime();
  return last_transaction;
}

auto Version::GetLastEditTransaction() -> EditTransaction& {
  if (_edit_transactions.empty()) {
    throw std::runtime_error("Version: No edit transaction to get");
  }
  return _edit_transactions.back();
}

}  // namespace puerhlab