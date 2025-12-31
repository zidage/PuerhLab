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

#include <xxhash.h>

#include <ctime>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "edit/history/edit_transaction.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "edit_transaction.hpp"
#include "type/hash_type.hpp"
#include "type/type.hpp"
#include "utils/id/id_generator.hpp"

namespace puerhlab {
using TxPos        = std::list<EditTransaction>::iterator;
using version_id_t = Hash128;

class Version {
 private:
  IncrID::IDGenerator<tx_id_t>       _tx_id_generator;
  static constexpr size_t            MAX_EDIT_TRANSACTIONS = 2048;
  /**
   * @brief Version ID (hash) for this version, calculated from building a merkle tree of all edit
   * transactions
   */
  version_id_t                       _version_id           = version_id_t{};

  version_id_t                       _parent_version_id    = version_id_t{};
  /**
   * @brief Last modified time for this version
   */
  std::time_t                        _added_time;
  std::time_t                        _last_modified_time;
  /**
   * @brief collection of images related to this version
   */
  sl_element_id_t                    _bound_image;
  /**
   * @brief Edit transactions for this edit version
   */
  std::list<EditTransaction>         _edit_transactions;

  std::unordered_map<tx_id_t, TxPos> _tx_id_map;

  std::shared_ptr<PipelineExecutor>  _base_pipeline_executor;

 public:
  Version();
  Version(sl_element_id_t bound_image);
  Version(sl_element_id_t bound_image, version_id_t parent_version_id);
  Version(nlohmann::json& j);

  void CalculateVersionID();
  auto GetVersionID() const -> version_id_t;

  void SetAddTime();
  auto GetAddTime() const -> std::time_t;
  void SetLastModifiedTime();
  auto GetLastModifiedTime() const -> std::time_t;

  void SetBoundImage(sl_element_id_t bound_image);
  auto GetBoundImage() const -> sl_element_id_t;

  void SetBasePipelineExecutor(std::shared_ptr<PipelineExecutor> pipeline_executor) {
    _base_pipeline_executor = pipeline_executor;
  }

  void AppendEditTransaction(EditTransaction&& edit_transaction);
  auto RemoveLastEditTransaction() -> EditTransaction;
  auto GetTransactionByID(tx_id_t transaction_id) -> EditTransaction&;
  auto GetLastEditTransaction() -> EditTransaction&;
  auto GetAllEditTransactions() const -> const std::list<EditTransaction>&;

  auto ToJSON() const -> nlohmann::json;
  void FromJSON(const nlohmann::json& j);
};
};  // namespace puerhlab