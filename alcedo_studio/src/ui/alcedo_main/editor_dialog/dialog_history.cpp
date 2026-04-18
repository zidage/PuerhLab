#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

auto EditorDialog::ReconstructPipelineParamsForVersion(Version& version) -> std::optional<nlohmann::json> {
    return versioning::ReconstructPipelineParamsForVersion(version, history_guard_);
  }

auto EditorDialog::ReloadUiStateFromPipeline(bool reset_to_defaults_if_missing) -> bool {
    const bool loaded = LoadStateFromPipelineIfPresent();
    if (!loaded && !reset_to_defaults_if_missing) {
      return false;
    }
    if (!loaded) {
      state_ = AdjustmentState{};
      SanitizeOdtStateForUi(state_.odt_);
      UpdateAllCdlWheelDerivedColors(state_);
      last_submitted_color_temp_request_.reset();
    } else {
      last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
    }
    committed_state_ = state_;
    SyncControlsFromState();
    TriggerQualityPreviewRenderFromPipeline();
    return true;
  }

auto EditorDialog::ApplyPipelineParamsToEditor(const nlohmann::json& params) -> bool {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return false;
    }

    auto exec = pipeline_guard_->pipeline_;
    exec->ImportPipelineParams(params);
    frame_manager_.AttachExecutionStages(exec);
    pipeline_guard_->dirty_ = true;
    last_applied_lut_path_.clear();

    return ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/true);
  }

auto EditorDialog::ReloadEditorFromHistoryVersion(Version& version, QString* error) -> bool {
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

void EditorDialog::CheckoutSelectedVersion(QListWidgetItem* item) {
    versioning::ResolvedVersionSelection selection{};
    QString                              selection_error;
    if (!versioning::ResolveSelectedVersion(item, history_guard_, &selection,
                                            &selection_error)) {
      if (!selection_error.isEmpty()) {
        QMessageBox::warning(this, Tr("History"), selection_error);
      }
      return;
    }

    QString reload_error;
    if (!selection.version || !ReloadEditorFromHistoryVersion(*selection.version, &reload_error)) {
      QMessageBox::warning(this, Tr("History"), reload_error);
      return;
    }

    working_version_ = versioning::SeedWorkingVersionFromCommit(
        element_id_, selection.version_id, pipeline_guard_,
        CurrentWorkingMode() != WorkingMode::Plain);
    UpdateVersionUi();
  }

void EditorDialog::UndoLastTransaction() {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return;
    }

    const auto undo_result = versioning::UndoLastTransaction(working_version_, pipeline_guard_);
    if (undo_result.no_transaction) {
      QMessageBox::information(this, Tr("History"), Tr("No transaction to undo."));
      return;
    }
    if (!undo_result.error.isEmpty()) {
      QMessageBox::warning(this, Tr("History"), undo_result.error);
      return;
    }
    if (!undo_result.undone) {
      return;
    }
    if (!ReloadUiStateFromPipeline(/*reset_to_defaults_if_missing=*/false)) {
      QMessageBox::warning(this, Tr("History"),
                           Tr("Undo failed while reloading pipeline state."));
      return;
    }
    UpdateVersionUi();
  }

void EditorDialog::UpdateVersionUi() {
    const versioning::VersionUiContext ui{
        .version_status     = version_status_,
        .commit_version_btn = commit_version_btn_,
        .undo_tx_btn        = undo_tx_btn_,
        .working_mode_combo = working_mode_combo_,
        .version_log        = version_log_,
        .tx_stack           = tx_stack_,
    };
    versioning::UpdateVersionUi(ui, working_version_, history_guard_,
                                [this]() { RefreshVersionLogSelectionStyles(); });
  }

void EditorDialog::CommitWorkingVersion() {
    const auto commit_result = versioning::CommitWorkingVersion(
        history_service_, history_guard_, pipeline_guard_, element_id_,
        std::move(working_version_));
    if (commit_result.no_transactions) {
      QMessageBox::information(this, Tr("History"), Tr("No uncommitted transactions."));
      return;
    }
    if (!commit_result.committed_id.has_value()) {
      QMessageBox::warning(this, Tr("History"),
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

auto EditorDialog::CurrentWorkingMode() const -> WorkingMode {
    return versioning::IsPlainModeSelected(working_mode_combo_) ? WorkingMode::Plain
                                                                : WorkingMode::Incremental;
  }

void EditorDialog::StartNewWorkingVersionFromUi() {
    working_version_ = versioning::SeedWorkingVersionFromUi(
        element_id_, history_guard_, pipeline_guard_,
        CurrentWorkingMode() == WorkingMode::Plain);
    UpdateVersionUi();
  }

void EditorDialog::StartNewWorkingVersionFromCommit(const Hash128& committed_id) {
    working_version_ = versioning::SeedWorkingVersionFromCommit(
        element_id_, committed_id, pipeline_guard_,
        CurrentWorkingMode() != WorkingMode::Plain);
  }

auto EditorDialog::ReadCurrentOperatorParams(PipelineStageName stage_name, OperatorType op_type) const
      -> std::optional<nlohmann::json> {
    if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
      return std::nullopt;
    }
    return pipeline_io::ReadCurrentOperatorParams(*pipeline_guard_->pipeline_, stage_name, op_type);
  }

std::pair<PipelineStageName, OperatorType> EditorDialog::FieldSpec(AdjustmentField field) const {
    return pipeline_io::FieldSpec(field);
  }

nlohmann::json EditorDialog::ParamsForField(AdjustmentField field, const AdjustmentState& s) const {
    return pipeline_io::ParamsForField(
        field, s, (pipeline_guard_ && pipeline_guard_->pipeline_)
                      ? pipeline_guard_->pipeline_.get()
                      : nullptr);
  }

bool EditorDialog::FieldChanged(AdjustmentField field) const {
    return pipeline_io::FieldChanged(field, state_, committed_state_);
  }

void EditorDialog::CommitAdjustment(AdjustmentField field) {
    if (!FieldChanged(field) || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
      // Still fulfill the "full res on release/change" behavior.
      ScheduleQualityPreviewRenderFromPipeline();
      return;
    }

    const auto [stage_name, op_type] = FieldSpec(field);
    const auto            old_params = ParamsForField(field, committed_state_);
    const auto            new_params = ParamsForField(field, state_);

    auto                  exec       = pipeline_guard_->pipeline_;
    auto&                 stage      = exec->GetStage(stage_name);
    const auto            op         = stage.GetOperator(op_type);
    const TransactionType tx_type =
        (op.has_value() && op.value() != nullptr) ? TransactionType::_EDIT : TransactionType::_ADD;

    EditTransaction tx{tx_type, op_type, stage_name, new_params};
    tx.SetLastOperatorParams(old_params);
    (void)tx.ApplyTransaction(*exec);

    working_version_.AppendEditTransaction(std::move(tx));
    pipeline_guard_->dirty_ = true;

    CopyFieldState(field, state_, committed_state_);
    if (field == AdjustmentField::CropRotate) {
      frame_manager_.MarkNeedsFullFramePreviewAfterGeometryCommit();
    }
    UpdateVersionUi();

    ScheduleQualityPreviewRenderFromPipeline();
  }
}  // namespace alcedo::ui
