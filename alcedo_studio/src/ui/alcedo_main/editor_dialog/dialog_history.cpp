#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

auto EditorDialog::ReconstructPipelineParamsForVersion(Version& version)
    -> std::optional<nlohmann::json> {
  return history_coordinator_
             ? history_coordinator_->ReconstructPipelineParamsForVersion(version)
             : std::nullopt;
}

auto EditorDialog::ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing) -> bool {
  return history_coordinator_ &&
         history_coordinator_->ReloadUiStateFromPipeline(reset_to_defaults_if_missing);
}

auto EditorDialog::ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool {
  return history_coordinator_ && history_coordinator_->ApplyPipelineParamsToEditor(params);
}

auto EditorDialog::ReloadEditorFromHistoryVersion(Version& version, QString* error) -> bool {
  return history_coordinator_ &&
         history_coordinator_->ReloadEditorFromHistoryVersion(version, error);
}

void EditorDialog::CheckoutSelectedVersion(QListWidgetItem* item) {
  if (history_coordinator_) {
    history_coordinator_->CheckoutSelectedVersion(item);
  }
}

void EditorDialog::CheckoutVersionById(const QString& version_id) {
  if (history_coordinator_) {
    history_coordinator_->CheckoutVersionById(version_id);
  }
}

void EditorDialog::UndoLastTransaction() {
  if (history_coordinator_) {
    history_coordinator_->UndoLastTransaction();
  }
}

void EditorDialog::UpdateVersionUi() {
  if (history_coordinator_) {
    history_coordinator_->UpdateVersionUi();
  }
}

void EditorDialog::CommitWorkingVersion() {
  if (history_coordinator_) {
    history_coordinator_->CommitWorkingVersion();
  }
}

void EditorDialog::StartNewWorkingVersionFromUi() {
  if (history_coordinator_) {
    history_coordinator_->StartNewWorkingVersionFromUi();
  }
}

void EditorDialog::StartNewWorkingVersionFromCommit(const Hash128& committed_id) {
  if (history_coordinator_) {
    history_coordinator_->StartNewWorkingVersionFromCommit(committed_id);
  }
}

auto EditorDialog::ReadCurrentOperatorParams(PipelineStageName stage_name,
                                             OperatorType      op_type) const
    -> std::optional<nlohmann::json> {
  if (!adjustment_session_) {
    return std::nullopt;
  }
  return adjustment_session_->ReadCurrentOperatorParams(stage_name, op_type);
}

std::pair<PipelineStageName, OperatorType> EditorDialog::FieldSpec(
    AdjustmentField field) const {
  return adjustment_session_ ? adjustment_session_->FieldSpec(field)
                             : pipeline_io::FieldSpec(field);
}

nlohmann::json EditorDialog::ParamsForField(AdjustmentField        field,
                                            const AdjustmentState& s) const {
  return adjustment_session_ ? adjustment_session_->ParamsForField(field, s)
                             : pipeline_io::ParamsForField(
                                   field, s,
                                   (pipeline_guard_ && pipeline_guard_->pipeline_)
                                       ? pipeline_guard_->pipeline_.get()
                                       : nullptr);
}

bool EditorDialog::FieldChanged(AdjustmentField field) const {
  return adjustment_session_ ? adjustment_session_->FieldChanged(field)
                             : pipeline_io::FieldChanged(field, state_, committed_state_);
}

void EditorDialog::CommitAdjustment(AdjustmentField field) {
  if (!adjustment_session_) {
    ScheduleQualityPreviewRenderFromPipeline();
    return;
  }
  const auto result = adjustment_session_->Commit(field);
  if (result.status == EditorAdjustmentSession::CommitStatus::Failed) {
    if (!result.error.isEmpty()) {
      QMessageBox::warning(this, Tr("Adjustment"),
                           Tr("Failed to apply adjustment: %1").arg(result.error));
    } else {
      QMessageBox::warning(this, Tr("Adjustment"), Tr("Failed to apply adjustment."));
    }
    return;
  }
}

}  // namespace alcedo::ui
