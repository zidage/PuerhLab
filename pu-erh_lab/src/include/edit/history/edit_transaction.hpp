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

#include <cstdint>
#include <optional>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "type/hash_type.hpp"

namespace puerhlab {
using tx_id_t = uint32_t;
enum class TransactionType : int { _ADD, _DELETE, _EDIT };
/**
 * @brief Represents a single edit transaction in the pipeline. Each transaction can be an addition,
 * deletion, or modification of an operator. Once created, transactions are immutable.
 *
 */
class EditTransaction {
 private:
  tx_id_t                       tx_id_;
  TransactionType               type_;

  OperatorType                  operator_type_;
  PipelineStageName             stage_name_;
  nlohmann::json                operator_params_;

  // Optional parent parameter to the last transaction that modified the same operator
  std::optional<nlohmann::json> last_operator_params_ = std::nullopt;

 public:
  EditTransaction(TransactionType type, OperatorType operator_type, PipelineStageName stage_name,
                  nlohmann::json         operator_params,
                  std::optional<tx_id_t> parent_tx_id = std::nullopt)
      : type_(type),
        operator_type_(operator_type),
        stage_name_(stage_name),
        operator_params_(operator_params) {
    // Compute a unique transaction ID based on the transaction details
    (void)parent_tx_id;
    auto params_str = operator_params_.dump();
  }

  EditTransaction(const nlohmann::json& j) { FromJSON(j); }

  auto SetTransactionID(tx_id_t id) { tx_id_ = id; }

  auto GetTransactionID() const -> tx_id_t { return tx_id_; }

  auto ApplyTransaction(PipelineExecutor& pipeline) -> bool;

  auto ToJSON() const -> nlohmann::json;
  void FromJSON(const nlohmann::json& j);

  auto GetTxOpStageName() const -> PipelineStageName { return stage_name_; }
  auto GetTxOperatorType() const -> OperatorType { return operator_type_; }

  void SetLastOperatorParams(const nlohmann::json& params) { last_operator_params_ = params; }
  auto GetLastOperatorParams() const -> std::optional<nlohmann::json> {
    return last_operator_params_;
  }

  auto UndoTransaction() -> EditTransaction {
    return EditTransaction(TransactionType::_EDIT, operator_type_, stage_name_,
                           last_operator_params_);
  }

  auto Hash() const -> Hash128 {
    std::string params_str = operator_params_.dump();
    Hash128     result =
        Hash128::Blend(Hash128::Blend(Hash128::Compute(&type_, sizeof(type_)),
                                      Hash128::Compute(&operator_type_, sizeof(operator_type_))),
                       Hash128::Blend(Hash128::Compute(&stage_name_, sizeof(stage_name_)),
                                      Hash128::Compute(&params_str, params_str.size())));
    if (last_operator_params_) {
      std::string last_params_str = last_operator_params_->dump();
      result = Hash128::Blend(result, Hash128::Compute(&last_params_str, last_params_str.size()));
    }
    return result;
  }
};
};  // namespace puerhlab