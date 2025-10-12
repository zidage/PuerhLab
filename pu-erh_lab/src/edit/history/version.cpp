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

#include <xxhash.h>

#include "edit/history/edit_transaction.hpp"
#include "type/hash_type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
Version::Version() : _tx_id_generator(0) {
  _added_time         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  _last_modified_time = _added_time;
  CalculateVersionID();
}

Version::Version(sl_element_id_t bound_image) : _tx_id_generator(0), _bound_image(bound_image) {
  _added_time         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  _last_modified_time = _added_time;
  CalculateVersionID();
}

Version::Version(sl_element_id_t bound_image, version_id_t parent_version_id)
    : _tx_id_generator(0), _bound_image(bound_image), _version_id(parent_version_id) {
  _added_time         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  _last_modified_time = _added_time;
  CalculateVersionID();
}

Version::Version(nlohmann::json& j) : _tx_id_generator(0) { FromJSON(j); }

void Version::CalculateVersionID() {
  _version_id = Hash128::Blend(_version_id,
                               Hash128::Compute(&_last_modified_time, sizeof(_last_modified_time)));
  if (!_edit_transactions.empty()) {
    const auto& tx = _edit_transactions.back();
    _version_id    = Hash128::Blend(_version_id, Hash128::Compute(&tx, sizeof(EditTransaction)));
  } else {
    // If there are no edit transactions, use the bound image ID to ensure uniqueness
    _version_id =
        Hash128::Blend(_version_id, Hash128::Compute(&_bound_image, sizeof(_bound_image)));
  }
}

auto Version::GetVersionID() const -> version_id_t { return Hash128(_version_id); }

auto Version::GetAddTime() const -> std::time_t { return _added_time; }

auto Version::GetLastModifiedTime() const -> std::time_t { return _last_modified_time; }

void Version::SetLastModifiedTime() {
  _last_modified_time = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

void Version::SetBoundImage(sl_element_id_t image_id) { _bound_image = image_id; }

auto Version::GetBoundImage() const -> sl_element_id_t { return _bound_image; }

void Version::AppendEditTransaction(EditTransaction&& edit_transaction) {
  if (_edit_transactions.size() >= MAX_EDIT_TRANSACTIONS) {
    auto removed_tx = std::move(_edit_transactions.front());
    _tx_id_map.erase(removed_tx.GetTransactionID());
    _edit_transactions.pop_front();
  }
  _edit_transactions.push_front(std::move(edit_transaction));

  // Assign a new transaction ID, update the ID map
  auto& last_tx = _edit_transactions.front();
  last_tx.SetTransactionID(_tx_id_generator.GenerateID());
  _tx_id_map[_edit_transactions.front().GetTransactionID()] = _edit_transactions.begin();

  // Check the same operator in the same stage, chain the parent transaction
  auto& stage = _base_pipeline_executor->GetStage(last_tx.GetTxOpStageName());
  auto  op    = stage.GetOperator(last_tx.GetTxOperatorType());
  // If this operator has been registered in this stage, chain the parent transaction
  if (op) {
    auto parent_params = (*op)->_op->GetParams();
    last_tx.SetLastOperatorParams(parent_params);
  }

  SetLastModifiedTime();
  CalculateVersionID();
}

auto Version::RemoveLastEditTransaction() -> EditTransaction {
  if (_edit_transactions.empty()) {
    throw std::runtime_error("Version: No edit transaction to remove");
  }
  EditTransaction last_transaction = std::move(_edit_transactions.front());
  _tx_id_map.erase(last_transaction.GetTransactionID());
  _edit_transactions.pop_front();
  SetLastModifiedTime();
  CalculateVersionID();
  return last_transaction;
}

auto Version::GetTransactionByID(tx_id_t transaction_id) -> EditTransaction& {
  auto it = _tx_id_map.find(transaction_id);
  if (it == _tx_id_map.end()) {
    throw std::runtime_error("Version: No edit transaction with the given ID");
  }
  return *(it->second);
}

auto Version::GetLastEditTransaction() -> EditTransaction& {
  if (_edit_transactions.empty()) {
    throw std::runtime_error("Version: No edit transaction to get");
  }
  return _edit_transactions.back();
}

auto Version::GetAllEditTransactions() const -> const std::list<EditTransaction>& {
  return _edit_transactions;
}

auto Version::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["version_id_low"]     = _version_id.low64();
  j["version_id_high"]    = _version_id.high64();
  j["added_time"]         = _added_time;
  j["last_modified_time"] = _last_modified_time;
  j["bound_image"]        = _bound_image;
  j["edit_transactions"]  = nlohmann::json::array();

  j["tx_id_start"]        = _tx_id_generator.GetCurrentID();
  for (const auto& tx : _edit_transactions) {
    j["edit_transactions"].push_back(tx.ToJSON());
  }

  return j;
}

void Version::FromJSON(const nlohmann::json& j) {
  if (!j.is_object() || !j.contains("version_id_low") || !j.contains("version_id_high") ||
      !j.contains("added_time") || !j.contains("last_modified_time") ||
      !j.contains("bound_image") || !j.contains("edit_transactions") ||
      !j.contains("tx_id_start")) {
    throw std::runtime_error("Version: Invalid JSON format");
  }
  _version_id =
      Hash128(j.at("version_id_low").get<uint64_t>(), j.at("version_id_high").get<uint64_t>());
  _added_time         = j.at("added_time").get<std::time_t>();
  _last_modified_time = j.at("last_modified_time").get<std::time_t>();
  _bound_image        = j.at("bound_image").get<sl_element_id_t>();
  _tx_id_generator.SetStartID(j.at("tx_id_start").get<tx_id_t>());
  _edit_transactions.clear();
  _tx_id_map.clear();
  for (const auto& tx_j : j.at("edit_transactions").get<nlohmann::json::array_t>()) {
    EditTransaction tx(tx_j);
    _edit_transactions.push_back(std::move(tx));
    _tx_id_map[_edit_transactions.back().GetTransactionID()] = std::prev(_edit_transactions.end());
  }
}
}  // namespace puerhlab