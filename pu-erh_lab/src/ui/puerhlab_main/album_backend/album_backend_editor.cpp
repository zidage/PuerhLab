auto AlbumBackend::BuildFilterNode(FilterOp joinOp) const -> BuildResult {
  std::optional<FilterNode> rules_node;
  std::vector<FilterNode>   conditions;

  for (const auto& rule : rule_model_.Rules()) {
    if (rule.value.trimmed().isEmpty()) {
      continue;
    }

    QString error;
    const auto value_opt = ParseFilterValue(rule.field, rule.value, error);
    if (!value_opt.has_value()) {
      return BuildResult{.node = std::nullopt, .error = error};
    }

    FieldCondition condition{
        .field_        = rule.field,
        .op_           = rule.op,
        .value_        = value_opt.value(),
        .second_value_ = std::nullopt,
    };

    if (rule.op == CompareOp::BETWEEN) {
      if (rule.value2.trimmed().isEmpty()) {
        return BuildResult{.node = std::nullopt, .error = "BETWEEN requires two values."};
      }
      const auto second_opt = ParseFilterValue(rule.field, rule.value2, error);
      if (!second_opt.has_value()) {
        return BuildResult{.node = std::nullopt, .error = error};
      }
      condition.second_value_ = second_opt.value();
    }

    conditions.push_back(FilterNode{
        FilterNode::Type::Condition, {}, {}, std::move(condition), std::nullopt});
  }

  if (!conditions.empty()) {
    if (conditions.size() == 1) {
      rules_node = conditions.front();
    } else {
      rules_node = FilterNode{
          FilterNode::Type::Logical, joinOp, std::move(conditions), {}, std::nullopt};
    }
  }

  if (rules_node.has_value()) {
    return BuildResult{.node = rules_node, .error = QString()};
  }
  return BuildResult{.node = std::nullopt, .error = QString()};
}

auto AlbumBackend::ParseFilterValue(FilterField field, const QString& text, QString& error) const
    -> std::optional<FilterValue> {
  const QString trimmed = text.trimmed();
  const auto    kind    = FilterRuleModel::KindForField(field);

  if (kind == FilterValueKind::String) {
    return FilterValue{trimmed.toStdWString()};
  }

  if (kind == FilterValueKind::Int64) {
    bool       ok = false;
    const auto v  = trimmed.toLongLong(&ok);
    if (!ok) {
      error = "Expected an integer value.";
      return std::nullopt;
    }
    return FilterValue{static_cast<int64_t>(v)};
  }

  if (kind == FilterValueKind::Double) {
    bool       ok = false;
    const auto v  = trimmed.toDouble(&ok);
    if (!ok) {
      error = "Expected a numeric value.";
      return std::nullopt;
    }
    return FilterValue{v};
  }

  const auto date_opt = ParseDate(trimmed);
  if (!date_opt.has_value()) {
    error = "Expected a date in YYYY-MM-DD format.";
    return std::nullopt;
  }
  return FilterValue{date_opt.value()};
}

auto AlbumBackend::ParseDate(const QString& text) -> std::optional<std::tm> {
  const QStringList parts = text.trimmed().split('-', Qt::SkipEmptyParts);
  if (parts.size() != 3) {
    return std::nullopt;
  }

  bool      ok_year = false;
  bool      ok_mon  = false;
  bool      ok_day  = false;
  const int year    = parts[0].toInt(&ok_year);
  const int month   = parts[1].toInt(&ok_mon);
  const int day     = parts[2].toInt(&ok_day);
  if (!ok_year || !ok_mon || !ok_day) {
    return std::nullopt;
  }

  const QDate date(year, month, day);
  if (!date.isValid()) {
    return std::nullopt;
  }

  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

bool AlbumBackend::IsImageInCurrentFolder(const AlbumItem& image) const {
  return image.parent_folder_id == current_folder_id_;
}

auto AlbumBackend::FormatFilterInfo(int shown, int total) const -> QString {
  if (total <= 0) {
    return "No images loaded.";
  }
  if (shown == total) {
    return QString("Showing %1 images").arg(total);
  }
  return QString("Showing %1 of %2").arg(shown).arg(total);
}

auto AlbumBackend::MakeThumbMap(const AlbumItem& image, int index) const -> QVariantMap {
  const QString aperture = image.aperture > 0.0 ? QString::number(image.aperture, 'f', 1) : "--";
  const QString focal    = image.focal_length > 0.0 ? QString::number(image.focal_length, 'f', 0) : "--";

  return QVariantMap{
      {"elementId", static_cast<uint>(image.element_id)},
      {"imageId", static_cast<uint>(image.image_id)},
      {"fileName", image.file_name.isEmpty() ? "(unnamed)" : image.file_name},
      {"cameraModel", image.camera_model.isEmpty() ? "Unknown" : image.camera_model},
      {"extension", image.extension.isEmpty() ? "--" : image.extension},
      {"iso", image.iso},
      {"aperture", aperture},
      {"focalLength", focal},
      {"captureDate", image.capture_date.isValid() ? image.capture_date.toString("yyyy-MM-dd") : "--"},
      {"rating", image.rating},
      {"tags", image.tags},
      {"accent", image.accent.isEmpty() ? AccentForIndex(static_cast<size_t>(index)) : image.accent},
      {"thumbUrl", image.thumb_data_url},
  };
}

void AlbumBackend::InitializeEditorLuts() {
  editor_lut_paths_.clear();
  editor_lut_options_.clear();

  editor_lut_paths_.push_back("");
  editor_lut_options_.push_back(QVariantMap{{"text", "None"}, {"value", 0}});

  // Prefer LUTs next to the executable (installed layout), fall back to source tree.
  const auto appLutsDir = std::filesystem::path(
      QCoreApplication::applicationDirPath().toStdWString()) / "LUTs";
  const auto srcLutsDir = std::filesystem::path(CONFIG_PATH) / "LUTs";
  const auto lutsDir    = std::filesystem::is_directory(appLutsDir) ? appLutsDir : srcLutsDir;
  const auto lutFiles   = ListCubeLutsInDir(lutsDir);
  for (const auto& path : lutFiles) {
    editor_lut_paths_.push_back(path.generic_string());
    editor_lut_options_.push_back(
        QVariantMap{{"text", QString::fromStdString(path.filename().string())},
                    {"value", static_cast<int>(editor_lut_paths_.size() - 1)}});
  }

  editor_lut_index_ = LutIndexForPath(editor_state_.lut_path_);
  if (!editor_lut_paths_.empty()) {
    editor_state_.lut_path_ = editor_lut_paths_[static_cast<size_t>(editor_lut_index_)];
  }
}

auto AlbumBackend::LutIndexForPath(const std::string& lutPath) const -> int {
  if (editor_lut_paths_.empty()) {
    return 0;
  }

  if (lutPath.empty()) {
    return 0;
  }

  for (size_t i = 0; i < editor_lut_paths_.size(); ++i) {
    if (editor_lut_paths_[i] == lutPath) {
      return static_cast<int>(i);
    }
  }

  const auto target = std::filesystem::path(lutPath).filename().wstring();
  for (size_t i = 0; i < editor_lut_paths_.size(); ++i) {
    if (std::filesystem::path(editor_lut_paths_[i]).filename().wstring() == target) {
      return static_cast<int>(i);
    }
  }
  return 0;
}

auto AlbumBackend::LoadEditorStateFromPipeline() -> bool {
  auto exec = editor_pipeline_guard_ ? editor_pipeline_guard_->pipeline_ : nullptr;
  if (!exec) {
    return false;
  }

  auto ReadFloat = [](const PipelineStage& stage, OperatorType type,
                      const char* key) -> std::optional<float> {
    const auto op = stage.GetOperator(type);
    if (!op.has_value() || op.value() == nullptr) {
      return std::nullopt;
    }
    const auto j = op.value()->ExportOperatorParams();
    if (!j.contains("params")) {
      return std::nullopt;
    }
    const auto& params = j["params"];
    if (!params.contains(key)) {
      return std::nullopt;
    }
    try {
      return params[key].get<float>();
    } catch (...) {
      return std::nullopt;
    }
  };

  auto ReadNestedFloat = [](const PipelineStage& stage, OperatorType type, const char* key1,
                            const char* key2) -> std::optional<float> {
    const auto op = stage.GetOperator(type);
    if (!op.has_value() || op.value() == nullptr) {
      return std::nullopt;
    }
    const auto j = op.value()->ExportOperatorParams();
    if (!j.contains("params")) {
      return std::nullopt;
    }
    const auto& params = j["params"];
    if (!params.contains(key1)) {
      return std::nullopt;
    }
    const auto& inner = params[key1];
    if (!inner.contains(key2)) {
      return std::nullopt;
    }
    try {
      return inner[key2].get<float>();
    } catch (...) {
      return std::nullopt;
    }
  };

  auto ReadString = [](const PipelineStage& stage, OperatorType type,
                       const char* key) -> std::optional<std::string> {
    const auto op = stage.GetOperator(type);
    if (!op.has_value() || op.value() == nullptr) {
      return std::nullopt;
    }
    const auto j = op.value()->ExportOperatorParams();
    if (!j.contains("params")) {
      return std::nullopt;
    }
    const auto& params = j["params"];
    if (!params.contains(key)) {
      return std::nullopt;
    }
    try {
      return params[key].get<std::string>();
    } catch (...) {
      return std::nullopt;
    }
  };

  const auto& basic  = exec->GetStage(PipelineStageName::Basic_Adjustment);
  const auto& color  = exec->GetStage(PipelineStageName::Color_Adjustment);
  const auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);

  // if (!basic.GetOperator(OperatorType::EXPOSURE).has_value()) {
  //   return false;
  // }

  EditorState loaded;
  if (const auto v = ReadFloat(basic, OperatorType::EXPOSURE, "exposure"); v.has_value()) {
    loaded.exposure_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::CONTRAST, "contrast"); v.has_value()) {
    loaded.contrast_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::BLACK, "black"); v.has_value()) {
    loaded.blacks_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::WHITE, "white"); v.has_value()) {
    loaded.whites_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::SHADOWS, "shadows"); v.has_value()) {
    loaded.shadows_ = v.value();
  }
  if (const auto v = ReadFloat(basic, OperatorType::HIGHLIGHTS, "highlights"); v.has_value()) {
    loaded.highlights_ = v.value();
  }
  if (const auto v = ReadFloat(color, OperatorType::SATURATION, "saturation"); v.has_value()) {
    loaded.saturation_ = v.value();
  }
  if (const auto v = ReadFloat(color, OperatorType::TINT, "tint"); v.has_value()) {
    loaded.tint_ = v.value();
  }
  if (const auto v = ReadNestedFloat(detail, OperatorType::SHARPEN, "sharpen", "offset");
      v.has_value()) {
    loaded.sharpen_ = v.value();
  }
  if (const auto v = ReadFloat(detail, OperatorType::CLARITY, "clarity"); v.has_value()) {
    loaded.clarity_ = v.value();
  }
  if (const auto lut = ReadString(color, OperatorType::LMT, "ocio_lmt");
      lut.has_value() && !lut->empty()) {
    loaded.lut_path_ = *lut;
  } else {
    loaded.lut_path_.clear();
  }

  editor_state_     = loaded;
  editor_lut_index_ = LutIndexForPath(editor_state_.lut_path_);
  if (!editor_lut_paths_.empty()) {
    editor_state_.lut_path_ = editor_lut_paths_[static_cast<size_t>(editor_lut_index_)];
  }

  return true;
}

void AlbumBackend::SetupEditorPipeline() {
  if (!editor_pipeline_guard_ || !editor_pipeline_guard_->pipeline_ || !project_) {
    throw std::runtime_error("Editor services are unavailable.");
  }

  auto imageDesc = project_->GetImagePoolService()->Read<std::shared_ptr<Image>>(
      editor_image_id_, [](const std::shared_ptr<Image>& img) { return img; });
  auto bytes = ByteBufferLoader::LoadFromImage(imageDesc);
  if (!bytes) {
    throw std::runtime_error("Failed to load image bytes.");
  }

  editor_base_task_                    = PipelineTask{};
  editor_base_task_.input_             = std::make_shared<ImageBuffer>(std::move(*bytes));
  editor_base_task_.pipeline_executor_ = editor_pipeline_guard_->pipeline_;
  editor_base_task_.options_.is_blocking_     = true;
  editor_base_task_.options_.is_callback_     = false;
  editor_base_task_.options_.is_seq_callback_ = false;
  editor_base_task_.options_.task_priority_   = 0;
  editor_base_task_.options_.render_desc_.render_type_ = RenderType::FAST_PREVIEW;

  auto exec = editor_pipeline_guard_->pipeline_;
  auto& loading = exec->GetStage(PipelineStageName::Image_Loading);

  nlohmann::json decodeParams;
#ifdef HAVE_CUDA
  decodeParams["raw"]["cuda"] = true;
#else
  decodeParams["raw"]["cuda"] = false;
#endif
  decodeParams["raw"]["highlights_reconstruct"] = true;
  decodeParams["raw"]["use_camera_wb"]          = true;
  decodeParams["raw"]["user_wb"]                = 7600.f;
  decodeParams["raw"]["backend"]                = "puerh";
  loading.SetOperator(OperatorType::RAW_DECODE, decodeParams);

  exec->SetExecutionStages();
}

void AlbumBackend::ApplyEditorStateToPipeline() {
  if (!editor_pipeline_guard_ || !editor_pipeline_guard_->pipeline_) {
    return;
  }

  auto exec          = editor_pipeline_guard_->pipeline_;
  auto& globalParams = exec->GetGlobalParams();

  auto& basic        = exec->GetStage(PipelineStageName::Basic_Adjustment);
  basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", editor_state_.exposure_}}, globalParams);
  basic.SetOperator(OperatorType::CONTRAST, {{"contrast", editor_state_.contrast_}}, globalParams);
  basic.SetOperator(OperatorType::BLACK, {{"black", editor_state_.blacks_}}, globalParams);
  basic.SetOperator(OperatorType::WHITE, {{"white", editor_state_.whites_}}, globalParams);
  basic.SetOperator(OperatorType::SHADOWS, {{"shadows", editor_state_.shadows_}}, globalParams);
  basic.SetOperator(OperatorType::HIGHLIGHTS, {{"highlights", editor_state_.highlights_}},
                    globalParams);

  auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
  color.SetOperator(OperatorType::SATURATION, {{"saturation", editor_state_.saturation_}},
                    globalParams);
  color.SetOperator(OperatorType::TINT, {{"tint", editor_state_.tint_}}, globalParams);
  color.SetOperator(OperatorType::LMT, {{"ocio_lmt", editor_state_.lut_path_}}, globalParams);

  auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
  detail.SetOperator(OperatorType::SHARPEN, {{"sharpen", {{"offset", editor_state_.sharpen_}}}},
                     globalParams);
  detail.SetOperator(OperatorType::CLARITY, {{"clarity", editor_state_.clarity_}}, globalParams);

  editor_pipeline_guard_->dirty_ = true;
}

void AlbumBackend::QueueEditorRender(RenderType renderType) {
  if (!editor_active_ || !editor_scheduler_ || !editor_pipeline_guard_) {
    return;
  }
  editor_pending_state_       = editor_state_;
  editor_pending_render_type_ = renderType;
  editor_has_pending_render_  = true;

  if (!editor_busy_) {
    editor_busy_ = true;
    emit EditorStateChanged();
  }

  if (!editor_render_inflight_) {
    StartNextEditorRender();
  }
}

void AlbumBackend::StartNextEditorRender() {
  if (!editor_has_pending_render_ || !editor_scheduler_ || !editor_pipeline_guard_ ||
      !editor_base_task_.pipeline_executor_) {
    return;
  }

  editor_has_pending_render_ = false;
  editor_state_              = editor_pending_state_;

  try {
    ApplyEditorStateToPipeline();
  } catch (...) {
    editor_status_ = "Failed to apply editor pipeline state.";
    editor_busy_   = false;
    emit EditorStateChanged();
    return;
  }

  PipelineTask task                       = editor_base_task_;
  task.options_.render_desc_.render_type_ = editor_pending_render_type_;
  task.options_.is_blocking_              = true;
  task.options_.is_callback_              = false;
  task.options_.is_seq_callback_          = false;
  task.options_.task_priority_            = 0;

  auto promise = std::make_shared<std::promise<std::shared_ptr<ImageBuffer>>>();
  auto future  = promise->get_future();
  task.result_ = std::move(promise);

  editor_render_inflight_ = true;
  editor_status_          = "Rendering preview...";
  emit EditorStateChanged();

  editor_scheduler_->ScheduleTask(std::move(task));
  editor_render_future_ = std::move(future);
  EnsureEditorPollTimer();
  if (editor_poll_timer_ && !editor_poll_timer_->isActive()) {
    editor_poll_timer_->start();
  }
}

void AlbumBackend::PollEditorRender() {
  if (!editor_render_future_.has_value()) {
    if (editor_poll_timer_ && editor_poll_timer_->isActive() && !editor_render_inflight_) {
      editor_poll_timer_->stop();
    }
    return;
  }

  if (editor_render_future_->wait_for(0ms) != std::future_status::ready) {
    return;
  }

  std::shared_ptr<ImageBuffer> result;
  try {
    result = editor_render_future_->get();
  } catch (...) {
    result.reset();
  }
  editor_render_future_.reset();
  editor_render_inflight_ = false;

  if (!UpdateEditorPreviewFromBuffer(result)) {
    editor_status_ = "Preview render did not produce an image.";
  } else {
    editor_status_ = "Preview ready.";
  }

  if (editor_has_pending_render_) {
    StartNextEditorRender();
    return;
  }

  editor_busy_ = false;
  emit EditorStateChanged();

  if (editor_poll_timer_ && editor_poll_timer_->isActive()) {
    editor_poll_timer_->stop();
  }
}

void AlbumBackend::EnsureEditorPollTimer() {
  if (editor_poll_timer_) {
    return;
  }
  editor_poll_timer_ = new QTimer(this);
  editor_poll_timer_->setInterval(16);
  connect(editor_poll_timer_, &QTimer::timeout, this, [this]() { PollEditorRender(); });
}

void AlbumBackend::FinalizeEditorSession(bool persistChanges) {
  if (!editor_pipeline_guard_) {
    editor_active_ = false;
    editor_busy_   = false;
    return;
  }

  if (editor_render_future_.has_value()) {
    try {
      editor_render_future_->wait();
      auto last = editor_render_future_->get();
      (void)UpdateEditorPreviewFromBuffer(last);
    } catch (...) {
    }
    editor_render_future_.reset();
  }

  editor_has_pending_render_ = false;
  editor_render_inflight_    = false;
  if (editor_poll_timer_ && editor_poll_timer_->isActive()) {
    editor_poll_timer_->stop();
  }

  const auto finishedElement = editor_element_id_;
  const auto finishedImage   = editor_image_id_;

  if (pipeline_service_) {
    try {
      if (persistChanges) {
        ApplyEditorStateToPipeline();
        editor_pipeline_guard_->dirty_ = true;
      } else {
        editor_pipeline_guard_->dirty_ = false;
      }
      pipeline_service_->SavePipeline(editor_pipeline_guard_);
      if (persistChanges) {
        pipeline_service_->Sync();
      }
    } catch (...) {
    }
  }

  if (persistChanges && project_) {
    try {
      project_->GetImagePoolService()->SyncWithStorage();
      project_->SaveProject(meta_path_);
    } catch (...) {
    }
  }

  if (persistChanges && thumbnail_service_ && finishedElement != 0 && finishedImage != 0) {
    try {
      thumbnail_service_->InvalidateThumbnail(finishedElement);
    } catch (...) {
    }
    if (IsThumbnailPinned(finishedElement)) {
      RequestThumbnail(finishedElement, finishedImage);
    } else {
      UpdateThumbnailDataUrl(finishedElement, QString());
    }
  }

  editor_pipeline_guard_.reset();
  editor_base_task_   = PipelineTask{};
  editor_active_      = false;
  editor_busy_        = false;
  editor_element_id_  = 0;
  editor_image_id_    = 0;
  editor_title_.clear();
  editor_status_      = persistChanges ? "Edits saved." : "Editor closed.";
  if (!editor_preview_url_.isEmpty()) {
    editor_preview_url_.clear();
    emit EditorPreviewChanged();
  }
  emit EditorStateChanged();
}

auto AlbumBackend::UpdateEditorPreviewFromBuffer(const std::shared_ptr<ImageBuffer>& buffer) -> bool {
  if (!buffer) {
    return false;
  }

  QString dataUrl;
  try {
    if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
      buffer->SyncToCPU();
    }
    if (!buffer->cpu_data_valid_) {
      return false;
    }

    QImage image = MatRgba32fToQImageCopy(buffer->GetCPUData());
    if (image.isNull()) {
      return false;
    }
    QImage scaled = image.scaled(1180, 760, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    dataUrl = DataUrlFromImage(scaled);
  } catch (...) {
    return false;
  }

  if (dataUrl.isEmpty()) {
    return false;
  }

  if (editor_preview_url_ != dataUrl) {
    editor_preview_url_ = dataUrl;
    emit EditorPreviewChanged();
  }
  return true;
}

void AlbumBackend::SetEditorAdjustment(float& field, double value, double minValue, double maxValue) {
  if (!editor_active_) {
    return;
  }
  const float clamped = ClampToRange(value, minValue, maxValue);
  if (NearlyEqual(field, clamped)) {
    return;
  }
  field = clamped;
  emit EditorStateChanged();
  QueueEditorRender(RenderType::FAST_PREVIEW);
}

void AlbumBackend::SetTaskState(const QString& status, int progress, bool cancelVisible) {
  task_status_         = status;
  task_progress_       = std::clamp(progress, 0, 100);
  task_cancel_visible_ = cancelVisible;
  emit TaskStateChanged();
}
