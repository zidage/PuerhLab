#include "edit/history/edit_transaction.hpp"

#include "edit/pipeline/pipeline.hpp"

namespace puerhlab {
auto EditTransaction::ApplyTransaction(PipelineExecutor& pipeline) -> bool {
  auto& stage = pipeline.GetStage(stage_name);
  switch (type) {
    case EditTransactionType::ADD:
      stage.SetOperator(operator_type, operator_params);
      return true;
    case EditTransactionType::DELETE:
      stage.EnableOperator(operator_type, false);
      return true;
    case EditTransactionType::EDIT:
      stage.SetOperator(operator_type, operator_params);
      return true;
  }
}

auto EditTransaction::RedoTransaction(PipelineExecutor& pipeline) -> bool {
  // Redoing a transaction is effectively the same as applying it again.
  return parent_transaction ? parent_transaction->ApplyTransaction(pipeline) : false;
}
};  // namespace puerhlab