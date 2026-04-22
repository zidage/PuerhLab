#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {
namespace {

constexpr float kDetailPreviewZoomEpsilon = 1.0e-4f;

}  // namespace

void EditorDialog::AdvancePreviewGeneration() {
  ++preview_generation_;
  detail_serial_                       = 0;
  latest_quality_base_generation_ready_ = 0;
  pending_fast_preview_request_.reset();
  pending_quality_base_render_request_.reset();
  if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
    quality_preview_timer_->stop();
  }
  if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
    fast_preview_submit_timer_->stop();
  }
  InvalidateDetailPreviewState();
}

void EditorDialog::InvalidateDetailPreviewState() {
  pending_detail_render_request_.reset();
  if (detail_preview_timer_ && detail_preview_timer_->isActive()) {
    detail_preview_timer_->stop();
  }
  if (viewer_) {
    viewer_->ClearExpectedDetailToken();
  }
}

auto EditorDialog::BuildPreviewMetadata(RenderType render_type) const -> FramePreviewMetadata {
  FramePreviewMetadata metadata{};
  metadata.preview_generation = preview_generation_;
  switch (render_type) {
    case RenderType::FAST_PREVIEW:
      metadata.frame_role = FrameRole::InteractivePrimary;
      break;
    case RenderType::QUALITY_BASE_PREVIEW:
      metadata.frame_role = FrameRole::QualityBase;
      break;
    case RenderType::DETAIL_ROI_PREVIEW:
      metadata.frame_role   = FrameRole::DetailPatch;
      metadata.detail_serial = detail_serial_;
      break;
    default:
      metadata.frame_role = FrameRole::QualityBase;
      break;
  }
  return metadata;
}

auto EditorDialog::IsDetailPreviewGeometryFallbackActive() const -> bool {
  return active_panel_ == ControlPanelKind::Geometry ||
         std::abs(state_.rotate_degrees_) > 1.0e-4f ||
         frame_manager_.NeedsFullFramePreviewAfterGeometryCommit();
}

auto EditorDialog::CanScheduleDetailPreview() const -> bool {
  if (!viewer_) {
    return false;
  }
  if (viewer_->GetViewZoom() <= (1.0f + kDetailPreviewZoomEpsilon)) {
    return false;
  }
  if (IsDetailPreviewGeometryFallbackActive()) {
    return false;
  }
  if (preview_generation_ == 0 ||
      latest_quality_base_generation_ready_ != preview_generation_) {
    return false;
  }
  const auto viewport_region = viewer_->GetViewportRenderRegion();
  return viewport_region.has_value();
}

void EditorDialog::MaybeScheduleDetailPreviewRenderFromViewport() {
  if (!CanScheduleDetailPreview()) {
    InvalidateDetailPreviewState();
    return;
  }
  ScheduleDetailPreviewRenderFromViewport();
}

void EditorDialog::TriggerQualityPreviewRenderFromPipeline() {
  if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
    quality_preview_timer_->stop();
  }

  AdjustmentState snapshot = state_;
  snapshot.type_           = RenderType::QUALITY_BASE_PREVIEW;
  EnqueueRenderRequest(snapshot, BuildPreviewMetadata(snapshot.type_),
                       /*apply_state=*/false,
                       /*use_viewport_region=*/false);
}

void EditorDialog::EnsureQualityPreviewTimer() {
  if (quality_preview_timer_) {
    return;
  }
  quality_preview_timer_ = new QTimer(this);
  quality_preview_timer_->setSingleShot(true);
  QObject::connect(quality_preview_timer_, &QTimer::timeout, this,
                   [this]() { TriggerQualityPreviewRenderFromPipeline(); });
}

void EditorDialog::EnsureDetailPreviewTimer() {
  if (detail_preview_timer_) {
    return;
  }
  detail_preview_timer_ = new QTimer(this);
  detail_preview_timer_->setSingleShot(true);
  QObject::connect(detail_preview_timer_, &QTimer::timeout, this,
                   [this]() { TriggerDetailPreviewRenderFromViewport(); });
}

void EditorDialog::ScheduleQualityPreviewRenderFromPipeline() {
  EnsureQualityPreviewTimer();
  quality_preview_timer_->start(static_cast<int>(kQualityPreviewDebounceInterval.count()));
}

void EditorDialog::ScheduleDetailPreviewRenderFromViewport() {
  EnsureDetailPreviewTimer();
  detail_preview_timer_->start(static_cast<int>(kViewportDetailDebounceInterval.count()));
}

void EditorDialog::TriggerDetailPreviewRenderFromViewport() {
  if (!CanScheduleDetailPreview() || !viewer_) {
    InvalidateDetailPreviewState();
    return;
  }

  ++detail_serial_;
  viewer_->SetExpectedDetailToken(preview_generation_, detail_serial_);

  AdjustmentState snapshot = state_;
  snapshot.type_           = RenderType::DETAIL_ROI_PREVIEW;
  EnqueueRenderRequest(snapshot, BuildPreviewMetadata(snapshot.type_),
                       /*apply_state=*/false,
                       /*use_viewport_region=*/true);
}

auto EditorDialog::CanSubmitFastPreviewNow() const -> bool {
  return controllers::render::CanSubmitFastPreviewNow(last_fast_preview_submit_time_,
                                                      std::chrono::steady_clock::now());
}

void EditorDialog::EnsureFastPreviewSubmitTimer() {
  if (fast_preview_submit_timer_) {
    return;
  }
  fast_preview_submit_timer_ = new QTimer(this);
  fast_preview_submit_timer_->setSingleShot(true);
  QObject::connect(fast_preview_submit_timer_, &QTimer::timeout, this, [this]() {
    if (!inflight_) {
      StartNext();
    }
  });
}

void EditorDialog::ArmFastPreviewSubmitTimer() {
  EnsureFastPreviewSubmitTimer();
  const int delay_ms = controllers::render::ComputeFastPreviewDelayMs(
      last_fast_preview_submit_time_, std::chrono::steady_clock::now());

  if (!fast_preview_submit_timer_->isActive()) {
    fast_preview_submit_timer_->start(delay_ms);
    return;
  }

  const int current_remaining = fast_preview_submit_timer_->remainingTime();
  if (current_remaining < 0 || delay_ms < current_remaining) {
    fast_preview_submit_timer_->start(delay_ms);
  }
}

void EditorDialog::EnqueueRenderRequest(const AdjustmentState& snapshot,
                                        const FramePreviewMetadata& frame_metadata,
                                        bool apply_state, bool use_viewport_region) {
  PendingRenderRequest request{snapshot, frame_metadata, apply_state, use_viewport_region};

  switch (snapshot.type_) {
    case RenderType::FAST_PREVIEW:
      pending_fast_preview_request_ = std::move(request);
      break;
    case RenderType::QUALITY_BASE_PREVIEW:
      pending_quality_base_render_request_ = std::move(request);
      break;
    case RenderType::DETAIL_ROI_PREVIEW:
      pending_detail_render_request_ = std::move(request);
      break;
    default:
      pending_quality_base_render_request_ = std::move(request);
      break;
  }

  if (!inflight_) {
    StartNext();
  }
}

void EditorDialog::RequestRender(bool use_viewport_region, bool bump_preview_generation) {
  if (bump_preview_generation) {
    AdvancePreviewGeneration();
  }

  AdjustmentState snapshot = state_;
  snapshot.type_           = RenderType::FAST_PREVIEW;
  EnqueueRenderRequest(snapshot, BuildPreviewMetadata(snapshot.type_),
                       /*apply_state=*/true, use_viewport_region);
}

void EditorDialog::RequestRenderWithoutApplyingState(bool use_viewport_region,
                                                     bool bump_preview_generation) {
  if (bump_preview_generation) {
    AdvancePreviewGeneration();
  }

  AdjustmentState snapshot = state_;
  snapshot.type_           = RenderType::FAST_PREVIEW;
  EnqueueRenderRequest(snapshot, BuildPreviewMetadata(snapshot.type_),
                       /*apply_state=*/false, use_viewport_region);
}

void EditorDialog::EnsurePollTimer() {
  if (poll_timer_) {
    return;
  }
  poll_timer_ = new QTimer(this);
  poll_timer_->setInterval(4);
  QObject::connect(poll_timer_, &QTimer::timeout, this, [this]() { PollInflight(); });
}

void EditorDialog::PollInflight() {
  if (!inflight_future_.has_value()) {
    if (poll_timer_ && poll_timer_->isActive() && !inflight_) {
      poll_timer_->stop();
    }
    return;
  }

  if (inflight_future_->wait_for(0ms) != std::future_status::ready) {
    return;
  }

  try {
    (void)inflight_future_->get();
  } catch (...) {
  }
  inflight_future_.reset();
  OnRenderFinished();
}

void EditorDialog::StartNext() {
  if (inflight_) {
    return;
  }

  std::optional<PendingRenderRequest> request;
  if (pending_quality_base_render_request_.has_value()) {
    request = pending_quality_base_render_request_;
    pending_quality_base_render_request_.reset();
  } else if (pending_detail_render_request_.has_value()) {
    request = pending_detail_render_request_;
    pending_detail_render_request_.reset();
  } else if (pending_fast_preview_request_.has_value()) {
    if (!CanSubmitFastPreviewNow()) {
      ArmFastPreviewSubmitTimer();
      return;
    }
    request = pending_fast_preview_request_;
    pending_fast_preview_request_.reset();
    last_fast_preview_submit_time_ = std::chrono::steady_clock::now();
    if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
      fast_preview_submit_timer_->stop();
    }
  }

  if (!request.has_value()) {
    return;
  }
  const PendingRenderRequest next_request = *request;

  if (spinner_) {
    spinner_->Start();
  }

  if (next_request.apply_state_) {
    ApplyStateToPipeline(next_request.state_);
    pipeline_guard_->dirty_ = true;
  }

  PipelineTask task                              = base_task_;
  task.options_.render_desc_.render_type_        = next_request.state_.type_;
  task.options_.render_desc_.use_viewport_region_ = next_request.use_viewport_region_;
  task.options_.render_desc_.frame_metadata_     = next_request.frame_metadata_;
  task.options_.is_callback_                     = false;
  task.options_.is_seq_callback_                 = false;
  task.options_.is_blocking_                     = true;

  auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
  auto fut     = promise->get_future();
  task.result_ = promise;

  inflight_        = true;
  inflight_request_ = next_request;
  scheduler_->ScheduleTask(std::move(task));

  inflight_future_ = std::move(fut);
  EnsurePollTimer();
  if (poll_timer_ && !poll_timer_->isActive()) {
    poll_timer_->start();
  }
}

void EditorDialog::OnRenderFinished() {
  inflight_ = false;

  if (spinner_) {
    spinner_->Stop();
  }

  const std::optional<PendingRenderRequest> finished_request = inflight_request_;
  inflight_request_.reset();

  if (RefreshColorTempRuntimeStateFromGlobalParams()) {
    SyncColorTempControlsFromState();
  }

  if (finished_request.has_value() &&
      finished_request->state_.type_ == RenderType::QUALITY_BASE_PREVIEW &&
      finished_request->frame_metadata_.preview_generation == preview_generation_) {
    latest_quality_base_generation_ready_ = preview_generation_;
    MaybeScheduleDetailPreviewRenderFromViewport();
  }

  if (pending_quality_base_render_request_.has_value() ||
      pending_detail_render_request_.has_value() ||
      pending_fast_preview_request_.has_value()) {
    StartNext();
  } else if (poll_timer_ && poll_timer_->isActive()) {
    poll_timer_->stop();
  }
}

}  // namespace alcedo::ui
