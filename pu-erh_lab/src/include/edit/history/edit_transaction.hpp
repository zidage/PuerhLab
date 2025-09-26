#pragma once

#include <optional>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "edit/pipeline/pipeline.hpp"
#include "type/hash_type.hpp"


namespace puerhlab {
using tx_id_t = Hash128;
enum class TransactionType : int { _ADD, _DELETE, _EDIT };
/**
 * @brief Represents a single edit transaction in the pipeline. Each transaction can be an addition,
 * deletion, or modification of an operator. Once created, transactions are immutable.
 *
 */
class EditTransaction {
 private:
  Hash128                _tx_id;
  TransactionType        _type;

  OperatorType           _operator_type;
  PipelineStageName      _stage_name;
  nlohmann::json         _operator_params;

  // Pointer to the previous transaction in the undo/redo chain.
  // This does not imply ownership; the lifetime of parent_transaction must be managed externally.
  // Used to traverse the history of edit transactions for undo/redo operations.
  std::optional<tx_id_t> _parent_tx_id = std::nullopt;

 public:
  EditTransaction(TransactionType type, OperatorType operator_type, PipelineStageName stage_name,
                  nlohmann::json         operator_params,
                  std::optional<tx_id_t> parent_tx_id = std::nullopt)
      : _type(type),
        _operator_type(operator_type),
        _stage_name(stage_name),
        _operator_params(operator_params),
        _parent_tx_id(parent_tx_id) {
    // Compute a unique transaction ID based on the transaction details
    auto params_str = _operator_params.dump();
    _tx_id          = Hash128::Blend(Hash128::Compute(&_operator_type, sizeof(_operator_type)),
                                     Hash128::Blend(Hash128::Compute(&_stage_name, sizeof(_stage_name)),
                                                    Hash128::Compute(&params_str, params_str.size())));
    if (_parent_tx_id) {
      _tx_id = Hash128::Blend(_tx_id, *_parent_tx_id);
    }
  }

  EditTransaction(const nlohmann::json& j) { FromJSON(j); }

  auto GetTransactionID() const -> Hash128 { return _tx_id; }

  auto ApplyTransaction(PipelineExecutor& pipeline) -> bool;

  auto ToJSON() const -> nlohmann::json;
  void FromJSON(const nlohmann::json& j);

  void SetParentTransaction(std::optional<tx_id_t> parent_id) {
    if (parent_id != _tx_id) {
      _parent_tx_id = parent_id;
    }
  }

  auto GetParentTransactionID() const -> Hash128 {
    if (_parent_tx_id) {
      return *_parent_tx_id;
    }
    return Hash128();  // Zero hash indicates no parent
  }

  auto UndoTransaction() -> EditTransaction {
    return EditTransaction(TransactionType::_DELETE, _operator_type, _stage_name, {},
                           _parent_tx_id);
  }
};
};  // namespace puerhlab