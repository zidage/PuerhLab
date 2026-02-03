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

#include "edit/history/version.hpp"

#include <xxhash.h>

#include <cstdint>

#include "edit/history/edit_transaction.hpp"
#include "type/hash_type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
Version::Version() : tx_id_generator_(0) {
  added_time_         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  last_modified_time_ = added_time_;
  CalculateVersionID();
}

Version::Version(sl_element_id_t bound_image) : tx_id_generator_(0), bound_image_(bound_image) {
  added_time_         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  last_modified_time_ = added_time_;
  CalculateVersionID();
}

Version::Version(sl_element_id_t bound_image, version_id_t parent_version_id)
    : tx_id_generator_(0), parent_version_id_(parent_version_id), bound_image_(bound_image) {
  added_time_         = std::chrono::system_clock::to_time_t(TimeProvider::Now());
  last_modified_time_ = added_time_;
  CalculateVersionID();
}

Version::Version(nlohmann::json& j) : tx_id_generator_(0) { FromJSON(j); }

void Version::CalculateVersionID() {
  Hash128 h = parent_version_id_;
  h         = Hash128::Blend(h, Hash128::Compute(&bound_image_, sizeof(bound_image_)));

  const uint64_t tx_count = static_cast<uint64_t>(edit_transactions_.size());
  h                      = Hash128::Blend(h, Hash128::Compute(&tx_count, sizeof(tx_count)));

  // edit_transactions_ is maintained as "newest first" (push_front), so iterate oldest->newest.
  for (auto it = edit_transactions_.rbegin(); it != edit_transactions_.rend(); ++it) {
    h = Hash128::Blend(h, it->Hash());
  }
  version_id_ = h;
}

auto Version::GetVersionID() const -> version_id_t { return version_id_; }

auto Version::GetParentVersionID() const -> version_id_t { return parent_version_id_; }

auto Version::HasParentVersion() const -> bool {
  return parent_version_id_.low64() != 0 || parent_version_id_.high64() != 0;
}

void Version::SetParentVersionID(version_id_t parent_version_id) {
  parent_version_id_ = parent_version_id;
  SetLastModifiedTime();
  CalculateVersionID();
}

void Version::ClearParentVersionID() {
  parent_version_id_ = version_id_t{};
  SetLastModifiedTime();
  CalculateVersionID();
}

auto Version::GetAddTime() const -> std::time_t { return added_time_; }

auto Version::GetLastModifiedTime() const -> std::time_t { return last_modified_time_; }

void Version::SetLastModifiedTime() {
  last_modified_time_ = std::chrono::system_clock::to_time_t(TimeProvider::Now());
}

void Version::SetBoundImage(sl_element_id_t image_id) { bound_image_ = image_id; }

auto Version::GetBoundImage() const -> sl_element_id_t { return bound_image_; }

void Version::AppendEditTransaction(EditTransaction&& edit_transaction) {
  if (edit_transactions_.size() >= MAX_EDIT_TRANSACTIONS) {
    auto removed_tx = std::move(edit_transactions_.front());
    tx_id_map_.erase(removed_tx.GetTransactionID());
    edit_transactions_.pop_front();
  }
  edit_transactions_.push_front(std::move(edit_transaction));

  // Assign a new transaction ID, update the ID map
  auto& last_tx = edit_transactions_.front();
  last_tx.SetTransactionID(tx_id_generator_.GenerateID());
  tx_id_map_[edit_transactions_.front().GetTransactionID()] = edit_transactions_.begin();

  // Check the same operator in the same stage, chain the parent transaction
  auto& stage = base_pipeline_executor_->GetStage(last_tx.GetTxOpStageName());
  auto  op    = stage.GetOperator(last_tx.GetTxOperatorType());
  // If this operator has been registered in this stage, chain the parent transaction
  if (!last_tx.GetLastOperatorParams().has_value() && op) {
    auto parent_params = (*op)->op_->GetParams();
    last_tx.SetLastOperatorParams(parent_params);
  }

  SetLastModifiedTime();
  CalculateVersionID();
}

auto Version::RemoveLastEditTransaction() -> EditTransaction {
  if (edit_transactions_.empty()) {
    throw std::runtime_error("Version: No edit transaction to remove");
  }
  EditTransaction last_transaction = std::move(edit_transactions_.front());
  tx_id_map_.erase(last_transaction.GetTransactionID());
  edit_transactions_.pop_front();
  SetLastModifiedTime();
  CalculateVersionID();
  return last_transaction;
}

auto Version::GetTransactionByID(tx_id_t transaction_id) -> EditTransaction& {
  auto it = tx_id_map_.find(transaction_id);
  if (it == tx_id_map_.end()) {
    throw std::runtime_error("Version: No edit transaction with the given ID");
  }
  return *(it->second);
}

auto Version::GetLastEditTransaction() -> EditTransaction& {
  if (edit_transactions_.empty()) {
    throw std::runtime_error("Version: No edit transaction to get");
  }
  return edit_transactions_.back();
}

auto Version::GetAllEditTransactions() const -> const std::list<EditTransaction>& {
  return edit_transactions_;
}

auto Version::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["version_id"]         = version_id_.ToString();
  j["version_id_low"]     = version_id_.low64();
  j["version_id_high"]    = version_id_.high64();
  j["parent_version_id"]  = parent_version_id_.ToString();
  j["parent_version_low"] = parent_version_id_.low64();
  j["parent_version_high"] = parent_version_id_.high64();
  j["added_time"]         = added_time_;
  j["last_modified_time"] = last_modified_time_;
  j["bound_image"]        = bound_image_;
  j["edit_transactions"]  = nlohmann::json::array();

  j["tx_id_start"]        = tx_id_generator_.GetCurrentID();
  for (const auto& tx : edit_transactions_) {
    j["edit_transactions"].push_back(tx.ToJSON());
  }

  return j;
}

void Version::FromJSON(const nlohmann::json& j) {
  if (!j.is_object() || (!j.contains("version_id_low") && !j.contains("version_id")) ||
      !j.contains("added_time") || !j.contains("last_modified_time") ||
      !j.contains("bound_image") || !j.contains("edit_transactions") ||
      !j.contains("tx_id_start")) {
    throw std::runtime_error("Version: Invalid JSON format");
  }
  if (j.contains("version_id_low") && j.contains("version_id_high")) {
    version_id_ =
        Hash128(j.at("version_id_low").get<uint64_t>(), j.at("version_id_high").get<uint64_t>());
  } else {
    version_id_ = Hash128::FromString(j.at("version_id").get<std::string>());
  }

  if (j.contains("parent_version_low") && j.contains("parent_version_high")) {
    parent_version_id_ = Hash128(j.at("parent_version_low").get<uint64_t>(),
                                 j.at("parent_version_high").get<uint64_t>());
  } else if (j.contains("parent_version_id")) {
    parent_version_id_ = Hash128::FromString(j.at("parent_version_id").get<std::string>());
  } else {
    parent_version_id_ = version_id_t{};
  }

  added_time_         = j.at("added_time").get<std::time_t>();
  last_modified_time_ = j.at("last_modified_time").get<std::time_t>();
  bound_image_        = j.at("bound_image").get<sl_element_id_t>();
  tx_id_generator_.SetStartID(j.at("tx_id_start").get<tx_id_t>());
  edit_transactions_.clear();
  tx_id_map_.clear();
  for (const auto& tx_j : j.at("edit_transactions").get<nlohmann::json::array_t>()) {
    EditTransaction tx(tx_j);
    edit_transactions_.push_back(std::move(tx));
    tx_id_map_[edit_transactions_.back().GetTransactionID()] = std::prev(edit_transactions_.end());
  }
}
}  // namespace puerhlab
