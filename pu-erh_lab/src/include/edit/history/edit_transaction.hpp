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
  tx_id_t                       _tx_id;
  TransactionType               _type;

  OperatorType                  _operator_type;
  PipelineStageName             _stage_name;
  nlohmann::json                _operator_params;

  // Optional parent parameter to the last transaction that modified the same operator
  std::optional<nlohmann::json> _last_operator_params = std::nullopt;

 public:
  EditTransaction(TransactionType type, OperatorType operator_type, PipelineStageName stage_name,
                  nlohmann::json         operator_params,
                  std::optional<tx_id_t> parent_tx_id = std::nullopt)
      : _type(type),
        _operator_type(operator_type),
        _stage_name(stage_name),
        _operator_params(operator_params) {
    // Compute a unique transaction ID based on the transaction details
    (void)parent_tx_id;
    auto params_str = _operator_params.dump();
  }

  EditTransaction(const nlohmann::json& j) { FromJSON(j); }

  auto SetTransactionID(tx_id_t id) { _tx_id = id; }

  auto GetTransactionID() const -> tx_id_t { return _tx_id; }

  auto ApplyTransaction(PipelineExecutor& pipeline) -> bool;

  auto ToJSON() const -> nlohmann::json;
  void FromJSON(const nlohmann::json& j);

  auto GetTxOpStageName() const -> PipelineStageName { return _stage_name; }
  auto GetTxOperatorType() const -> OperatorType { return _operator_type; }

  void SetLastOperatorParams(const nlohmann::json& params) { _last_operator_params = params; }
  auto GetLastOperatorParams() const -> std::optional<nlohmann::json> {
    return _last_operator_params;
  }

  auto UndoTransaction() -> EditTransaction {
    return EditTransaction(TransactionType::_EDIT, _operator_type, _stage_name,
                           _last_operator_params);
  }

  auto Hash() const -> Hash128 {
    std::string params_str = _operator_params.dump();
    Hash128     result =
        Hash128::Blend(Hash128::Blend(Hash128::Compute(&_type, sizeof(_type)),
                                      Hash128::Compute(&_operator_type, sizeof(_operator_type))),
                       Hash128::Blend(Hash128::Compute(&_stage_name, sizeof(_stage_name)),
                                      Hash128::Compute(&params_str, params_str.size())));
    if (_last_operator_params) {
      std::string last_params_str = _last_operator_params->dump();
      result = Hash128::Blend(result, Hash128::Compute(&last_params_str, last_params_str.size()));
    }
    return result;
  }
};
};  // namespace puerhlab