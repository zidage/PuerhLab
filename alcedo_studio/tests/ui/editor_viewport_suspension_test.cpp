//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

/// @file editor_viewport_suspension_test.cpp
/// @brief A minimized editor window suspends its viewport. Frames that finish
/// rendering while it is minimized must be refused at once with a typed
/// "presentation suspended" mapping, never by waiting for a scene-graph consumer
/// that a minimized window does not provide; restoring the window must make the
/// same target writable again.

#include <gtest/gtest.h>

#include <QSignalSpy>
#include <QTest>
#include <chrono>
#include <future>
#include <thread>

#include "ui/edit_viewer/frame_sink.hpp"
#include "ui/editor_rhi/direct_frame_sink.hpp"
#include "ui/editor_rhi/direct_present_queue.hpp"
#include "ui/editor_rhi/editor_backend.hpp"
#include "ui/editor_rhi/editor_viewport_item.hpp"
#include "ui/editor_rhi/frame_presentation_lease.hpp"

namespace alcedo::editor_rhi {
namespace {

using namespace std::chrono_literals;

constexpr std::uint64_t kImageIdentity = 42;
constexpr std::uint64_t kSessionEpoch  = 1;
// The old producer path waited 5 s for a consumer, twice per frame.
constexpr auto          kFailFastLimit = 1s;

class EditorViewportSuspensionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Shared targets (CUDA/OpenCL) are the path that maps a write target.
    SetActiveEditorBackend(EditorBackend::Cuda);
    viewport_ = std::make_unique<EditorViewportItem>();
    viewport_->setImageIdentity(kImageIdentity);
    viewport_->setSessionEpoch(kSessionEpoch);
    // Stands in for EditorViewportRenderer::synchronize() on a shown window.
    viewport_->resumePresentation();
    queue().SetConsumerAvailable(true);
  }

  auto queue() -> DirectPresentQueue& { return *viewport_->present_queue(); }
  auto sink() -> DirectFrameSink& { return *viewport_->frameSink(); }

  // Publishes a target as the render thread does after the first frame, so the
  // next frame of this size reuses it without a scene-graph handshake.
  auto PublishTarget(int width, int height) -> int {
    const auto prepared = queue().PrepareWrite(width, height, kSessionEpoch, kImageIdentity);
    EXPECT_TRUE(prepared.ok);
    DirectPresentQueue::SlotNative native{};
    native.backend           = EditorBackend::Cuda;
    native.handle_kind       = LeaseNativeHandleKind::D3D11Texture2D;
    native.writable_kind     = LeaseWritableResourceKind::CudaArray;
    native.native_handle     = 0x1000;
    native.writable_resource = 0x2000;
    EXPECT_TRUE(queue().PublishCreatedSlot(prepared.slot_index, width, height, native,
                                           kSessionEpoch, kImageIdentity));
    return prepared.slot_index;
  }

  // Runs one presenter-side frame (bind, EnsureSize, map) on a pipeline worker
  // thread, as the CUDA/OpenCL presenters do.
  auto MapOnWorker(int width, int height) -> std::future<FrameWriteMapping> {
    return std::async(std::launch::async, [this, width, height] {
      FramePreviewMetadata metadata{};
      metadata.frame_role = FrameRole::QualityBase;
      sink().BindFrameSubmission({metadata, FramePresentationMode::ViewportTransformed});
      sink().EnsureSize(width, height);
      return sink().MapResourceForWrite(FrameMemoryDomain::CudaDevice);
    });
  }

  std::unique_ptr<EditorViewportItem> viewport_;
};

TEST_F(EditorViewportSuspensionTest, MinimizedViewportRefusesFramesAtOnceAndResumesWritable) {
  const int slot = PublishTarget(64, 48);
  ASSERT_GE(slot, 0);

  QSignalSpy suspended_spy(viewport_.get(), &EditorViewportItem::PresentationSuspendedChanged);
  // What the window's visibilityChanged(Minimized) handler does.
  viewport_->suspendPresentation();
  EXPECT_TRUE(viewport_->presentationSuspended());
  EXPECT_TRUE(queue().PresentationSuspended());
  EXPECT_EQ(suspended_spy.count(), 1);

  // A frame for the existing target and a frame that needs a new target both
  // finish rendering while the window is minimized.
  const auto started  = std::chrono::steady_clock::now();
  auto       existing = MapOnWorker(64, 48);
  ASSERT_EQ(existing.wait_for(kFailFastLimit), std::future_status::ready);
  const auto existing_mapping = existing.get();
  auto       resized          = MapOnWorker(128, 96);
  ASSERT_EQ(resized.wait_for(kFailFastLimit), std::future_status::ready);
  const auto resized_mapping = resized.get();
  EXPECT_LT(std::chrono::steady_clock::now() - started, kFailFastLimit);

  EXPECT_FALSE(static_cast<bool>(existing_mapping));
  EXPECT_TRUE(existing_mapping.presentation_suspended);
  EXPECT_FALSE(static_cast<bool>(resized_mapping));
  EXPECT_TRUE(resized_mapping.presentation_suspended);
  // A minimized window queues no target request for its render thread.
  EXPECT_TRUE(queue().DrainSizeRequests().empty());
  // The existing target is refused while suspended.
  EXPECT_FALSE(queue().BeginWrite(slot).has_value());

  // Restore: visibilityChanged(Windowed) resumes, then synchronize() turns the
  // consumer back on.
  viewport_->resumePresentation();
  EXPECT_FALSE(viewport_->presentationSuspended());
  EXPECT_FALSE(queue().PresentationSuspended());
  EXPECT_EQ(suspended_spy.count(), 2);
  queue().SetConsumerAvailable(true);

  // The same target that failed with "BeginWrite failed after reserve" before
  // is writable again.
  const auto begun = queue().BeginWrite(slot);
  ASSERT_TRUE(begun.has_value());
  EXPECT_EQ(begun->width, 64);
  queue().AbandonWrite(slot);
}

TEST_F(EditorViewportSuspensionTest, MinimizeReleasesAProducerWaitingForANewTarget) {
  // The window is shown, but its render thread has not created the requested
  // target yet: the producer waits in the native-target handshake.
  auto waiting = MapOnWorker(256, 192);
  const bool requested = QTest::qWaitFor([this] { return queue().HasReadyFrame(); }, 2000);
  ASSERT_TRUE(requested);
  EXPECT_EQ(waiting.wait_for(50ms), std::future_status::timeout);

  // Minimizing must end that wait. The old handshake kept waiting (with the
  // render lock held) until the window was restored.
  viewport_->suspendPresentation();
  const auto status = waiting.wait_for(kFailFastLimit);
  if (status != std::future_status::ready) {
    queue().Shutdown();  // Release the worker so the test can report the failure.
  }
  ASSERT_EQ(status, std::future_status::ready);
  const auto mapping = waiting.get();
  EXPECT_FALSE(static_cast<bool>(mapping));
  EXPECT_TRUE(mapping.presentation_suspended);
}

TEST_F(EditorViewportSuspensionTest, SuspendAndResumeSignalOnlyOnChange) {
  QSignalSpy suspended_spy(viewport_.get(), &EditorViewportItem::PresentationSuspendedChanged);
  viewport_->resumePresentation();
  EXPECT_EQ(suspended_spy.count(), 0);
  viewport_->suspendPresentation();
  viewport_->suspendPresentation();
  EXPECT_EQ(suspended_spy.count(), 1);
  viewport_->resumePresentation();
  viewport_->resumePresentation();
  EXPECT_EQ(suspended_spy.count(), 2);
}

}  // namespace
}  // namespace alcedo::editor_rhi
