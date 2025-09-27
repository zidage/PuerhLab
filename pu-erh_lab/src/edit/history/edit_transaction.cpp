#include "edit/history/edit_transaction.hpp"

#include "edit/pipeline/pipeline.hpp"

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
  j["id"]              = _tx_id;
  j["type"]            = static_cast<int>(_type);
  j["operator_type"]   = static_cast<int>(_operator_type);
  j["stage_name"]      = static_cast<int>(_stage_name);
  j["operator_params"] = _operator_params;
  if (_last_operator_params) {
    j["last_operator_params"] = *_last_operator_params;
  }

  return j;
}

void EditTransaction::FromJSON(const nlohmann::json& j) {
  _tx_id           = j["id"].get<tx_id_t>();
  _type            = static_cast<TransactionType>(j["type"].get<int>());
  _operator_type   = static_cast<OperatorType>(j["operator_type"].get<int>());
  _stage_name      = static_cast<PipelineStageName>(j["stage_name"].get<int>());
  _operator_params = j["operator_params"];
  if (j.contains("last_operator_params")) {
    _last_operator_params = j["last_operator_params"];
  } else {
    _last_operator_params = std::nullopt;
  }
}

};  // namespace puerhlab