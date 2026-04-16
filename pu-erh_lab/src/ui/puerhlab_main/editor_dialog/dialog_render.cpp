#include "ui/puerhlab_main/editor_dialog/dialog_internal.hpp"

namespace puerhlab::ui {

void EditorDialog::TriggerQualityPreviewRenderFromPipeline() {
    if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
      quality_preview_timer_->stop();
    }
    state_.type_ = RenderType::FULL_RES_PREVIEW;
    RequestRenderWithoutApplyingState();
    state_.type_ = RenderType::FAST_PREVIEW;
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

void EditorDialog::ScheduleQualityPreviewRenderFromPipeline() {
    EnsureQualityPreviewTimer();
    quality_preview_timer_->start(static_cast<int>(kQualityPreviewDebounceInterval.count()));
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

void EditorDialog::EnqueueRenderRequest(const AdjustmentState& snapshot, bool apply_state,
                            bool use_viewport_region) {
    PendingRenderRequest request{snapshot, apply_state, use_viewport_region};

    if (snapshot.type_ == RenderType::FAST_PREVIEW) {
      // Industry pattern for interactive rendering:
      // coalesce rapid slider updates and keep only the newest fast preview.
      if (quality_preview_timer_ && quality_preview_timer_->isActive()) {
        quality_preview_timer_->stop();
      }
      pending_fast_preview_request_ = std::move(request);
    } else {
      // Keep quality requests ordered and drop stale fast previews.
      pending_quality_render_requests_.push_back(std::move(request));
      pending_fast_preview_request_.reset();
      if (fast_preview_submit_timer_ && fast_preview_submit_timer_->isActive()) {
        fast_preview_submit_timer_->stop();
      }
    }

    if (!inflight_) {
      StartNext();
    }
  }

void EditorDialog::RequestRender(bool use_viewport_region) {
    EnqueueRenderRequest(state_, true, use_viewport_region);
  }

void EditorDialog::RequestRenderWithoutApplyingState(bool use_viewport_region) {
    EnqueueRenderRequest(state_, false, use_viewport_region);
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
    if (!pending_quality_render_requests_.empty()) {
      request = pending_quality_render_requests_.front();
      pending_quality_render_requests_.pop_front();
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

    PipelineTask task                       = base_task_;
    task.options_.render_desc_.render_type_ = next_request.state_.type_;
    task.options_.render_desc_.use_viewport_region_ = next_request.use_viewport_region_;
    task.options_.is_callback_              = false;
    task.options_.is_seq_callback_          = false;
    task.options_.is_blocking_              = true;

    auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
    auto fut     = promise->get_future();
    task.result_ = promise;

    inflight_    = true;
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

    if (RefreshColorTempRuntimeStateFromGlobalParams()) {
      SyncColorTempControlsFromState();
    }

    if (!pending_quality_render_requests_.empty() || pending_fast_preview_request_.has_value()) {
      StartNext();
    } else if (poll_timer_ && poll_timer_->isActive()) {
      poll_timer_->stop();
    }
  }
}  // namespace puerhlab::ui
