//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/history/editor_history_coordinator.hpp"

#include <utility>

#include <QMessageBox>

#include "ui/alcedo_main/editor_dialog/controllers/history_controller.hpp"
#include "ui/alcedo_main/i18n.hpp"

namespace alcedo::ui {

EditorHistoryCoordinator::EditorHistoryCoordinator(Dependencies dependencies, Callbacks callbacks)
    : dependencies_(std::move(dependencies)), callbacks_(std::move(callbacks)) {}

auto EditorHistoryCoordinator::WorkingVersion() -> Version& { return working_version_; }

auto EditorHistoryCoordinator::WorkingVersion() const -> const Version& {
  return working_version_;
}

void EditorHistoryCoordinator::SetUiContext(const versioning::VersionUiContext& ui) {
  ui_ = ui;
}

void EditorHistoryCoordinator::SeedWorkingVersionFromLatest() {
  working_version_ =
      controllers::SeedWorkingVersionFromLatest(dependencies_.element_id,
                                                dependencies_.history_guard);
  if (dependencies_.pipeline_guard && dependencies_.pipeline_guard->pipeline_) {
    working_version_.SetBasePipelineExecutor(dependencies_.pipeline_guard->pipeline_);
  }
}

auto EditorHistoryCoordinator::ReconstructPipelineParamsForVersion(Version& version) const
    -> std::optional<nlohmann::json> {
  return versioning::ReconstructPipelineParamsForVersion(version, dependencies_.history_guard);
}

auto EditorHistoryCoordinator::ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing)
    -> bool {
  return callbacks_.reload_ui_state_from_pipeline
             ? callbacks_.reload_ui_state_from_pipeline(reset_to_defaults_if_missing)
             : false;
}

auto EditorHistoryCoordinator::ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool {
  if (!dependencies_.pipeline_guard || !dependencies_.pipeline_guard->pipeline_) {
    return false;
  }

  auto exec = dependencies_.pipeline_guard->pipeline_;
  exec->ImportPipelineParams(params);
  dependencies_.pipeline_guard->dirty_ = true;
  if (callbacks_.after_pipeline_params_imported) {
    callbacks_.after_pipeline_params_imported();
  }

  return ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/true);
}

auto EditorHistoryCoordinator::ReloadEditorFromHistoryVersion(Version& version, QString* error)
    -> bool {
  const auto selected_params = ReconstructPipelineParamsForVersion(version);
  if (!selected_params.has_value()) {
    if (error) {
      *error = Tr("Could not reconstruct pipeline params for the selected version.");
    }
    return false;
  }

  if (!ApplyPipelineParamsToEditor(*selected_params)) {
    if (error) {
      *error = Tr("Failed to apply selected version to the editor.");
    }
    return false;
  }
  return true;
}

void EditorHistoryCoordinator::CheckoutSelectedVersion(QListWidgetItem* item) {
  versioning::ResolvedVersionSelection selection{};
  QString                              selection_error;
  if (!versioning::ResolveSelectedVersion(item, dependencies_.history_guard, &selection,
                                          &selection_error)) {
    if (!selection_error.isEmpty()) {
      QMessageBox::warning(dependencies_.message_parent, Tr("History"), selection_error);
    }
    return;
  }

  QString reload_error;
  if (!selection.version || !ReloadEditorFromHistoryVersion(*selection.version, &reload_error)) {
    QMessageBox::warning(dependencies_.message_parent, Tr("History"), reload_error);
    return;
  }

  working_version_ = versioning::SeedWorkingVersionFromCommit(
      dependencies_.element_id, selection.version_id, dependencies_.pipeline_guard,
      IsIncrementalWorkingMode());
  UpdateVersionUi();
}

void EditorHistoryCoordinator::UndoLastTransaction() {
  if (!dependencies_.pipeline_guard || !dependencies_.pipeline_guard->pipeline_) {
    return;
  }

  const auto undo_result =
      versioning::UndoLastTransaction(working_version_, dependencies_.pipeline_guard);
  if (undo_result.no_transaction) {
    QMessageBox::information(dependencies_.message_parent, Tr("History"),
                             Tr("No transaction to undo."));
    return;
  }
  if (!undo_result.error.isEmpty()) {
    QMessageBox::warning(dependencies_.message_parent, Tr("History"), undo_result.error);
    return;
  }
  if (!undo_result.undone) {
    return;
  }
  if (!ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/false)) {
    QMessageBox::warning(dependencies_.message_parent, Tr("History"),
                         Tr("Undo failed while reloading pipeline state."));
    return;
  }
  UpdateVersionUi();
}

void EditorHistoryCoordinator::UpdateVersionUi() {
  versioning::UpdateVersionUi(ui_, working_version_, dependencies_.history_guard,
                              callbacks_.refresh_version_log_selection_styles);
}

void EditorHistoryCoordinator::CommitWorkingVersion() {
  const auto commit_result = versioning::CommitWorkingVersion(
      dependencies_.history_service, dependencies_.history_guard, dependencies_.pipeline_guard,
      dependencies_.element_id, std::move(working_version_));
  if (commit_result.no_transactions) {
    QMessageBox::information(dependencies_.message_parent, Tr("History"),
                             Tr("No uncommitted transactions."));
    return;
  }
  if (!commit_result.committed_id.has_value()) {
    QMessageBox::warning(dependencies_.message_parent, Tr("History"),
                         commit_result.error.isEmpty() ? Tr("Commit failed.")
                                                       : commit_result.error);
    if (commit_result.recovery_working_version.has_value()) {
      working_version_ = std::move(*commit_result.recovery_working_version);
      UpdateVersionUi();
    }
    return;
  }

  StartNewWorkingVersionFromCommit(*commit_result.committed_id);
  UpdateVersionUi();
}

void EditorHistoryCoordinator::StartNewWorkingVersionFromUi() {
  working_version_ = versioning::SeedWorkingVersionFromUi(
      dependencies_.element_id, dependencies_.history_guard, dependencies_.pipeline_guard,
      IsPlainWorkingMode());
  UpdateVersionUi();
}

void EditorHistoryCoordinator::StartNewWorkingVersionFromCommit(const Hash128& committed_id) {
  working_version_ = versioning::SeedWorkingVersionFromCommit(
      dependencies_.element_id, committed_id, dependencies_.pipeline_guard,
      IsIncrementalWorkingMode());
}

auto EditorHistoryCoordinator::IsPlainWorkingMode() const -> bool {
  return callbacks_.is_plain_working_mode ? callbacks_.is_plain_working_mode() : false;
}

auto EditorHistoryCoordinator::IsIncrementalWorkingMode() const -> bool {
  return !IsPlainWorkingMode();
}

}  // namespace alcedo::ui
