#pragma once

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline.hpp"

namespace puerhlab {
enum class EditTransactionType { ADD, DELETE, EDIT };
/**
 * @brief Represents a single edit transaction in the pipeline. Each transaction can be an addition,
 * deletion, or modification of an operator. Once created, transactions are immutable.
 *
 */
class EditTransaction {
 private:
  int                 transaction_id;
  EditTransactionType type;

  OperatorType        operator_type;
  PipelineStageName   stage_name;
  nlohmann::json      operator_params;

  // Pointer to the previous transaction in the undo/redo chain.
  // This does not imply ownership; the lifetime of parent_transaction must be managed externally.
  // Used to traverse the history of edit transactions for undo/redo operations.
  EditTransaction*    parent_transaction;

 public:
  EditTransaction(int transaction_id, EditTransactionType type, OperatorType operator_type,
                  PipelineStageName stage_name, nlohmann::json operator_params,
                  EditTransaction* parent_transaction = nullptr)
      : transaction_id(transaction_id),
        type(type),
        operator_type(operator_type),
        stage_name(stage_name),
        operator_params(operator_params),
        parent_transaction(parent_transaction == this ? nullptr : parent_transaction) {}

  auto ApplyTransaction(PipelineExecutor& pipeline) -> bool;
  auto RedoTransaction(PipelineExecutor& pipeline) -> bool;
};
};  // namespace puerhlab