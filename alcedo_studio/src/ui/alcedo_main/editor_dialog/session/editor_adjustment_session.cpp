//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/session/editor_adjustment_session.hpp"

#include <exception>
#include <utility>

#include "edit/history/edit_transaction.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"

namespace alcedo::ui {

EditorAdjustmentSession::EditorAdjustmentSession(Dependencies dependencies, Callbacks callbacks)
    : dependencies_(std::move(dependencies)), callbacks_(std::move(callbacks)) {}

void EditorAdjustmentSession::Preview(const AdjustmentPreview& preview) { last_preview_ = preview; }

auto EditorAdjustmentSession::Commit(AdjustmentField field) -> CommitResult {
  return Commit(AdjustmentCommit{.field = field});
}

auto EditorAdjustmentSession::Commit(const AdjustmentCommit& commit) -> CommitResult {
  if (commit.policy != CommitPolicy::AppendTransaction || !FieldChanged(commit.field) ||
      !HasPipeline() || dependencies_.working_version == nullptr ||
      dependencies_.state == nullptr || dependencies_.committed_state == nullptr) {
    ScheduleQualityPreview();
    return {.status = CommitStatus::UnchangedOrUnavailable};
  }

  const auto [stage_name, op_type] = FieldSpec(commit.field);
  const auto old_params =
      commit.old_params.value_or(ParamsForField(commit.field, *dependencies_.committed_state));
  const auto new_params =
      commit.new_params.value_or(ParamsForField(commit.field, *dependencies_.state));

  auto                  exec  = dependencies_.pipeline_guard->pipeline_;
  auto&                 stage = exec->GetStage(stage_name);
  const auto            op    = stage.GetOperator(op_type);
  const TransactionType tx_type =
      (op.has_value() && op.value() != nullptr) ? TransactionType::_EDIT : TransactionType::_ADD;

  EditTransaction tx{tx_type, op_type, stage_name, new_params};
  tx.SetLastOperatorParams(old_params);
  try {
    (void)tx.ApplyTransaction(*exec);
  } catch (const std::exception& e) {
    return {.status = CommitStatus::Failed, .error = QString::fromUtf8(e.what())};
  } catch (...) {
    return {.status = CommitStatus::Failed};
  }

  dependencies_.working_version->AppendEditTransaction(std::move(tx));
  dependencies_.pipeline_guard->dirty_ = true;

  CopyFieldState(commit.field, *dependencies_.state, *dependencies_.committed_state);
  if (commit.field == AdjustmentField::CropRotate &&
      callbacks_.mark_full_frame_preview_after_geometry_commit) {
    callbacks_.mark_full_frame_preview_after_geometry_commit();
  }
  if (callbacks_.update_version_ui) {
    callbacks_.update_version_ui();
  }

  if (callbacks_.advance_preview_generation) {
    callbacks_.advance_preview_generation();
  }
  ScheduleQualityPreview();

  return {.status = CommitStatus::Applied};
}

auto EditorAdjustmentSession::LoadFromPipeline() -> bool {
  if (!HasPipeline() || dependencies_.state == nullptr) {
    return false;
  }

  auto [loaded_state, has_loaded_any] = pipeline_io::LoadStateFromPipeline(
      *dependencies_.pipeline_guard->pipeline_, *dependencies_.state);
  if (!has_loaded_any) {
    return false;
  }
  *dependencies_.state = std::move(loaded_state);
  if (dependencies_.committed_state != nullptr) {
    *dependencies_.committed_state = *dependencies_.state;
  }
  return true;
}

auto EditorAdjustmentSession::ReloadFromImportedPipelineParams() -> bool {
  return LoadFromPipeline();
}

auto EditorAdjustmentSession::ReadCurrentOperatorParams(PipelineStageName stage_name,
                                                        OperatorType      op_type) const
    -> std::optional<nlohmann::json> {
  if (!HasPipeline()) {
    return std::nullopt;
  }
  return pipeline_io::ReadCurrentOperatorParams(*dependencies_.pipeline_guard->pipeline_,
                                                stage_name, op_type);
}

auto EditorAdjustmentSession::FieldSpec(AdjustmentField field) const
    -> std::pair<PipelineStageName, OperatorType> {
  return pipeline_io::FieldSpec(field);
}

auto EditorAdjustmentSession::ParamsForField(AdjustmentField        field,
                                             const AdjustmentState& state) const -> nlohmann::json {
  return pipeline_io::ParamsForField(
      field, state, HasPipeline() ? dependencies_.pipeline_guard->pipeline_.get() : nullptr);
}

auto EditorAdjustmentSession::FieldChanged(AdjustmentField field) const -> bool {
  if (dependencies_.state == nullptr || dependencies_.committed_state == nullptr) {
    return false;
  }
  return pipeline_io::FieldChanged(field, *dependencies_.state, *dependencies_.committed_state);
}

auto EditorAdjustmentSession::HasPipeline() const -> bool {
  return dependencies_.pipeline_guard && dependencies_.pipeline_guard->pipeline_;
}

void EditorAdjustmentSession::ScheduleQualityPreview() const {
  if (callbacks_.schedule_quality_preview) {
    callbacks_.schedule_quality_preview();
  }
}

}  // namespace alcedo::ui
