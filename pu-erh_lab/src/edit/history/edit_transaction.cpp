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

auto EditTransaction::RedoTransaction(PipelineExecutor& pipeline) -> bool {
  // Redoing a transaction is effectively the same as applying it again.
  return _parent_transaction ? _parent_transaction->ApplyTransaction(pipeline) : false;
}

auto EditTransaction::ToJSON() const -> nlohmann::json {
  nlohmann::json j;
  j["id"]                    = _transaction_id;
  j["type"]                  = static_cast<int>(_type);
  j["operator_type"]         = static_cast<int>(_operator_type);
  j["stage_name"]            = static_cast<int>(_stage_name);
  j["operator_params"]       = _operator_params;
  // j["parent_transaction_id"] = _parent_transaction ? _parent_transaction->_transaction_id : -1;

  return j;
}

void EditTransaction::FromJSON(const nlohmann::json& j) {
  _transaction_id     = j["id"].get<int>();
  _type               = static_cast<TransactionType>(j["type"].get<int>());
  _operator_type      = static_cast<OperatorType>(j["operator_type"].get<int>());
  _stage_name         = static_cast<PipelineStageName>(j["stage_name"].get<int>());
  _operator_params    = j["operator_params"];
  _parent_transaction = nullptr;
}


};  // namespace puerhlab