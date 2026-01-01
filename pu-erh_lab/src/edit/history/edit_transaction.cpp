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

#include "edit/history/edit_transaction.hpp"

#include "edit/pipeline/pipeline.hpp"

namespace puerhlab {
auto EditTransaction::ApplyTransaction(PipelineExecutor& pipeline) -> bool {
  auto& stage = pipeline.GetStage(stage_name_);
  switch (type_) {
    case TransactionType::_ADD:
      stage.SetOperator(operator_type_, operator_params_);
      return true;
    case TransactionType::_DELETE:
      stage.EnableOperator(operator_type_, false);
      return true;
    case TransactionType::_EDIT:
      stage.SetOperator(operator_type_, operator_params_);
      return true;
  }
}

auto EditTransaction::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["id"]              = tx_id_;
  j["type"]            = static_cast<int>(type_);
  j["operator_type"]   = static_cast<int>(operator_type_);
  j["stage_name"]      = static_cast<int>(stage_name_);
  j["operator_params"] = operator_params_;
  if (last_operator_params_) {
    j["last_operator_params"] = *last_operator_params_;
  }

  return j;
}

void EditTransaction::FromJSON(const nlohmann::json& j) {
  tx_id_           = j["id"].get<tx_id_t>();
  type_            = static_cast<TransactionType>(j["type"].get<int>());
  operator_type_   = static_cast<OperatorType>(j["operator_type"].get<int>());
  stage_name_      = static_cast<PipelineStageName>(j["stage_name"].get<int>());
  operator_params_ = j["operator_params"];
  if (j.contains("last_operator_params")) {
    last_operator_params_ = j["last_operator_params"];
  } else {
    last_operator_params_ = std::nullopt;
  }
}

};  // namespace puerhlab