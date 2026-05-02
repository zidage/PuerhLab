#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

void EditorDialog::AdvancePreviewGeneration() {
  if (render_coordinator_) {
    render_coordinator_->AdvancePreviewGeneration();
  }
}

void EditorDialog::InvalidateDetailPreviewState() {
  if (render_coordinator_) {
    render_coordinator_->InvalidateDetailPreviewState();
  }
}

auto EditorDialog::BuildPreviewMetadata(RenderType render_type) const -> FramePreviewMetadata {
  return render_coordinator_ ? render_coordinator_->BuildPreviewMetadata(render_type)
                             : FramePreviewMetadata{};
}

auto EditorDialog::IsDetailPreviewGeometryFallbackActive() const -> bool {
  return render_coordinator_ && render_coordinator_->IsDetailPreviewGeometryFallbackActive();
}

auto EditorDialog::CanScheduleDetailPreview() const -> bool {
  return render_coordinator_ && render_coordinator_->CanScheduleDetailPreview();
}

void EditorDialog::MaybeScheduleDetailPreviewRenderFromViewport() {
  if (render_coordinator_) {
    render_coordinator_->MaybeScheduleDetailPreviewRenderFromViewport();
  }
}

void EditorDialog::TriggerQualityPreviewRenderFromPipeline() {
  if (render_coordinator_) {
    render_coordinator_->TriggerQualityPreviewRenderFromPipeline();
  }
}

void EditorDialog::EnsureQualityPreviewTimer() {
  if (render_coordinator_) {
    render_coordinator_->EnsureQualityPreviewTimer();
  }
}

void EditorDialog::EnsureDetailPreviewTimer() {
  if (render_coordinator_) {
    render_coordinator_->EnsureDetailPreviewTimer();
  }
}

void EditorDialog::ScheduleQualityPreviewRenderFromPipeline() {
  if (render_coordinator_) {
    render_coordinator_->ScheduleQualityPreviewRenderFromPipeline();
  }
}

void EditorDialog::ScheduleDetailPreviewRenderFromViewport() {
  if (render_coordinator_) {
    render_coordinator_->ScheduleDetailPreviewRenderFromViewport();
  }
}

void EditorDialog::TriggerDetailPreviewRenderFromViewport() {
  if (render_coordinator_) {
    render_coordinator_->TriggerDetailPreviewRenderFromViewport();
  }
}

auto EditorDialog::CanSubmitFastPreviewNow() const -> bool {
  return render_coordinator_ && render_coordinator_->CanSubmitFastPreviewNow();
}

void EditorDialog::EnsureFastPreviewSubmitTimer() {
  if (render_coordinator_) {
    render_coordinator_->EnsureFastPreviewSubmitTimer();
  }
}

void EditorDialog::ArmFastPreviewSubmitTimer() {
  if (render_coordinator_) {
    render_coordinator_->ArmFastPreviewSubmitTimer();
  }
}

void EditorDialog::EnqueueRenderRequest(const AdjustmentState& snapshot,
                                        const FramePreviewMetadata& frame_metadata,
                                        bool apply_state, bool use_viewport_region) {
  if (render_coordinator_) {
    render_coordinator_->EnqueueRenderRequest(snapshot, frame_metadata, apply_state,
                                              use_viewport_region);
  }
}

void EditorDialog::RequestRender(bool use_viewport_region, bool bump_preview_generation) {
  if (render_coordinator_) {
    render_coordinator_->RequestRender(use_viewport_region, bump_preview_generation);
  }
}

void EditorDialog::RequestRenderWithoutApplyingState(bool use_viewport_region,
                                                     bool bump_preview_generation) {
  if (render_coordinator_) {
    render_coordinator_->RequestRenderWithoutApplyingState(use_viewport_region,
                                                           bump_preview_generation);
  }
}

void EditorDialog::EnsurePollTimer() {
  if (render_coordinator_) {
    render_coordinator_->EnsurePollTimer();
  }
}

void EditorDialog::PollInflight() {
  if (render_coordinator_) {
    render_coordinator_->PollInflight();
  }
}

void EditorDialog::StartNext() {
  if (render_coordinator_) {
    render_coordinator_->StartNext();
  }
}

void EditorDialog::OnRenderFinished() {
  if (render_coordinator_) {
    render_coordinator_->OnRenderFinished();
  }
}

}  // namespace alcedo::ui
