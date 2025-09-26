#include "edit/history/edit_transaction.hpp"

#include "edit/pipeline/pipeline.hpp"
#include "type/hash_type.hpp"

namespace puerhlab {
auto EditTransaction::ApplyTransaction(PipelineExecutor& pipeline) -> bool {
  auto& stage = pipeline.GetStage(_stage_name);
  switch (_type) {
    case TransactionType::_ADD:
      stage.SetOperator(_operator_type, _operator_params);
      return true;
    case TransactionType::_DELETE:
      stage.EnableOperator(_operator_type, false);
      return true;
    case TransactionType::_EDIT:
      stage.SetOperator(_operator_type, _operator_params);
      return true;
  }
}

auto EditTransaction::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["id"]                    = _tx_id.ToString();
  j["type"]                  = static_cast<int>(_type);
  j["operator_type"]         = static_cast<int>(_operator_type);
  j["stage_name"]            = static_cast<int>(_stage_name);
  j["operator_params"]       = _operator_params;
  j["parent_tx_id"] = _parent_tx_id ? _parent_tx_id->ToString() : "";

  return j;
}

void EditTransaction::FromJSON(const nlohmann::json& j) {
  _tx_id     = Hash128::FromString(j["id"].get<std::string>());
  _type               = static_cast<TransactionType>(j["type"].get<int>());
  _operator_type      = static_cast<OperatorType>(j["operator_type"].get<int>());
  _stage_name         = static_cast<PipelineStageName>(j["stage_name"].get<int>());
  _operator_params    = j["operator_params"];
  if (j.contains("parent_tx_id") && !j["parent_tx_id"].get<std::string>().empty()) {
    _parent_tx_id = Hash128::FromString(j["parent_tx_id"].get<std::string>());
  } else {
    _parent_tx_id = std::nullopt;
  }
}


};  // namespace puerhlab