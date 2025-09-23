#pragma once

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline.hpp"

namespace puerhlab {
enum class TransactionType : int { _ADD, _DELETE, _EDIT };
/**
 * @brief Represents a single edit transaction in the pipeline. Each transaction can be an addition,
 * deletion, or modification of an operator. Once created, transactions are immutable.
 *
 */
class EditTransaction {
 private:
  int               _transaction_id;
  TransactionType   _type;

  OperatorType      _operator_type;
  PipelineStageName _stage_name;
  nlohmann::json    _operator_params;

  // Pointer to the previous transaction in the undo/redo chain.
  // This does not imply ownership; the lifetime of parent_transaction must be managed externally.
  // Used to traverse the history of edit transactions for undo/redo operations.
  EditTransaction*  _parent_transaction;

 public:
  EditTransaction(int transaction_id, TransactionType type, OperatorType operator_type,
                  PipelineStageName stage_name, nlohmann::json operator_params,
                  EditTransaction* _parent_transaction = nullptr)
      : _transaction_id(transaction_id),
        _type(type),
        _operator_type(operator_type),
        _stage_name(stage_name),
        _operator_params(operator_params),
        _parent_transaction(_parent_transaction == this ? nullptr : _parent_transaction) {}

  EditTransaction(const nlohmann::json& j) {
    FromJSON(j);
  }

  auto GetTransactionID() const -> int { return _transaction_id; }

  auto ApplyTransaction(PipelineExecutor& pipeline) -> bool;
  auto RedoTransaction(PipelineExecutor& pipeline) -> bool;

  auto ToJSON() const -> nlohmann::json;
  void FromJSON(const nlohmann::json& j);

  void SetParentTransaction(EditTransaction* parent) {
    if (parent != this) {
      _parent_transaction = parent;
    }
  }
};
};  // namespace puerhlab