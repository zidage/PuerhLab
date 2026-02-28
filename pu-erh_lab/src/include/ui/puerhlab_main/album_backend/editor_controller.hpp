#pragma once

#include <QTimer>
#include <QVariantList>

#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ui/puerhlab_main/album_backend/album_types.hpp"
#include "app/pipeline_service.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "renderer/pipeline_task.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Manages the inline image editor (pipeline-based adjustments + preview).
class EditorController {
 public:
  explicit EditorController(AlbumBackend& backend);

  void OpenEditor(sl_element_id_t elementId, image_id_t imageId);
  void CloseEditor();
  void ResetEditorAdjustments();
  void RequestEditorFullPreview();
  void SetEditorLutIndex(int index);
  void SetEditorExposure(double value);
  void SetEditorContrast(double value);
  void SetEditorSaturation(double value);
  void SetEditorTint(double value);
  void SetEditorBlacks(double value);
  void SetEditorWhites(double value);
  void SetEditorShadows(double value);
  void SetEditorHighlights(double value);
  void SetEditorSharpen(double value);
  void SetEditorClarity(double value);
  void InitializeEditorLuts();
  void FinalizeEditorSession(bool persistChanges);

  [[nodiscard]] bool editor_active() const { return editor_active_; }
  [[nodiscard]] bool editor_busy() const { return editor_busy_; }
  [[nodiscard]] auto editor_element_id() const -> sl_element_id_t { return editor_element_id_; }
  [[nodiscard]] auto editor_image_id() const -> image_id_t { return editor_image_id_; }
  [[nodiscard]] auto editor_title() const -> const QString& { return editor_title_; }
  [[nodiscard]] auto editor_status() const -> const QString& { return editor_status_; }
  [[nodiscard]] auto editor_preview_url() const -> const QString& { return editor_preview_url_; }
  [[nodiscard]] auto editor_lut_options() const -> const QVariantList& { return editor_lut_options_; }
  [[nodiscard]] int  editor_lut_index() const { return editor_lut_index_; }
  [[nodiscard]] auto editor_state() const -> const EditorState& { return editor_state_; }

 private:
  int  LutIndexForPath(const std::string& lutPath) const;
  bool LoadEditorStateFromPipeline();
  void SetupEditorPipeline();
  void ApplyEditorStateToPipeline();
  void QueueEditorRender(RenderType renderType);
  void StartNextEditorRender();
  void PollEditorRender();
  void EnsureEditorPollTimer();
  bool UpdateEditorPreviewFromBuffer(const std::shared_ptr<ImageBuffer>& buffer);
  void SetEditorAdjustment(float& field, double value, double minValue, double maxValue);

  AlbumBackend& backend_;

  bool            editor_active_      = false;
  bool            editor_busy_        = false;
  sl_element_id_t editor_element_id_  = 0;
  image_id_t      editor_image_id_    = 0;
  QString         editor_title_{};
  QString         editor_status_      = "Select a photo to edit.";
  QString         editor_preview_url_{};
  QVariantList    editor_lut_options_{};
  std::vector<std::string> editor_lut_paths_{};
  int             editor_lut_index_   = 0;
  EditorState     editor_state_{};
  EditorState     editor_initial_state_{};
  EditorState     editor_pending_state_{};
  RenderType      editor_pending_render_type_ = RenderType::FAST_PREVIEW;
  bool            editor_has_pending_render_  = false;
  bool            editor_render_inflight_     = false;
  std::shared_ptr<PipelineGuard>     editor_pipeline_guard_{};
  std::shared_ptr<PipelineScheduler> editor_scheduler_{};
  PipelineTask    editor_base_task_{};
  QTimer*         editor_poll_timer_  = nullptr;
  std::optional<std::future<std::shared_ptr<ImageBuffer>>> editor_render_future_{};
};

}  // namespace puerhlab::ui
