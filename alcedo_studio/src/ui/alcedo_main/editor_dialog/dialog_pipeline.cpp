#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

bool EditorDialog::LoadStateFromPipelineIfPresent() {
    auto exec = pipeline_guard_ ? pipeline_guard_->pipeline_ : nullptr;
    if (!exec) {
      return false;
    }
    auto [loaded_state, has_loaded_any] = pipeline_io::LoadStateFromPipeline(*exec, state_);
    if (!has_loaded_any) {
      return false;
    }
    SanitizeOdtStateForUi(loaded_state.odt_);
    if (loaded_state.color_temp_mode_ == ColorTempMode::AS_SHOT &&
        loaded_state.color_temp_supported_) {
      CacheAsShotColorTemp(loaded_state.color_temp_resolved_cct_,
                           loaded_state.color_temp_resolved_tint_);
    }
    state_ = loaded_state;
    last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
    return true;
  }

void EditorDialog::SetupPipeline() {
    base_task_.input_             = controllers::LoadImageInputBuffer(image_pool_, image_id_);
    base_task_.pipeline_executor_ = pipeline_guard_->pipeline_;

    auto           exec           = pipeline_guard_->pipeline_;
    controllers::EnsureLoadingOperatorDefaults(exec);
    frame_manager_.AttachExecutionStages(exec);
    frame_manager_.SyncViewerDisplayEncoding(state_.odt_.encoding_space_,
                                             state_.odt_.encoding_eotf_);

    // Inject pre-extracted raw metadata from the Image so downstream operators
    // (ColorTemp, LensCalib) resolve eagerly.
    try {
      auto img = image_pool_->Read<std::shared_ptr<Image>>(
          image_id_, [](const std::shared_ptr<Image>& i) { return i; });
      if (img && img->HasRawColorContext()) {
        exec->InjectRawMetadata(img->GetRawColorContext());
        WarmAsShotColorTempCacheFromRawMetadata();
      }
    } catch (...) {
      // Non-fatal: metadata injection is best-effort.
    }

    // Cached pipelines can clear transient GPU resources when returned to the service.
    // PipelineMgmtService now resyncs global params on load, so we no longer need a
    // per-dialog LMT rebind hack here.
    last_applied_lut_path_.clear();
  }

void EditorDialog::ApplyStateToPipeline(const AdjustmentState& render_state) {
    AdjustmentState render_state_sanitized = render_state;
    SanitizeOdtStateForUi(render_state_sanitized.odt_);

    auto  exec          = pipeline_guard_->pipeline_;
    auto& global_params = exec->GetGlobalParams();
    auto& loading       = exec->GetStage(PipelineStageName::Image_Loading);
    auto& geometry      = exec->GetStage(PipelineStageName::Geometry_Adjustment);
    auto& to_ws         = exec->GetStage(PipelineStageName::To_WorkingSpace);
    auto& output        = exec->GetStage(PipelineStageName::Output_Transform);

    loading.SetOperator(OperatorType::RAW_DECODE,
                        ParamsForField(AdjustmentField::RawDecode, render_state_sanitized));
    loading.SetOperator(OperatorType::LENS_CALIBRATION,
                        ParamsForField(AdjustmentField::LensCalib, render_state_sanitized),
                        global_params);
    loading.EnableOperator(OperatorType::LENS_CALIBRATION,
                           render_state_sanitized.lens_calib_enabled_, global_params);

    const auto color_temp_request = BuildColorTempRequest(render_state_sanitized);
    const bool color_temp_missing = !to_ws.GetOperator(OperatorType::COLOR_TEMP).has_value();
    if (color_temp_missing || !last_submitted_color_temp_request_.has_value() ||
        !ColorTempRequestEqual(*last_submitted_color_temp_request_, color_temp_request)) {
      to_ws.SetOperator(OperatorType::COLOR_TEMP,
                        ParamsForField(AdjustmentField::ColorTemp, render_state_sanitized),
                        global_params);
      to_ws.EnableOperator(OperatorType::COLOR_TEMP, true, global_params);
      last_submitted_color_temp_request_ = color_temp_request;
    } else {
      to_ws.EnableOperator(OperatorType::COLOR_TEMP, true, global_params);
    }

    output.SetOperator(OperatorType::ODT,
                       ParamsForField(AdjustmentField::Odt, render_state_sanitized),
                       global_params);
    output.EnableOperator(OperatorType::ODT, true, global_params);

    // Geometry editing is overlay-only. While the geometry panel is active,
    // render the full pre-geometry frame so recropping can always expand back
    // to the original image bounds.
    nlohmann::json crop_rotate_params;
    bool           apply_crop = committed_state_.crop_enabled_;
    if (active_panel_ == ControlPanelKind::Geometry) {
      crop_rotate_params = pipeline_defaults::MakeDefaultCropRotateParams();
      crop_rotate_params["crop_rotate"]["enable_crop"]    = false;
      crop_rotate_params["crop_rotate"]["expand_to_fit"]  = committed_state_.crop_expand_to_fit_;
      apply_crop = false;
    } else {
      crop_rotate_params = ParamsForField(AdjustmentField::CropRotate, committed_state_);
    }

    crop_rotate_params["crop_rotate"]["enable_crop"] = apply_crop;
    const bool geometry_enabled = crop_rotate_params["crop_rotate"].value("enabled", false);
    geometry.SetOperator(OperatorType::CROP_ROTATE, crop_rotate_params, global_params);
    geometry.EnableOperator(OperatorType::CROP_ROTATE, geometry_enabled, global_params);

    auto& basic         = exec->GetStage(PipelineStageName::Basic_Adjustment);
    basic.SetOperator(OperatorType::EXPOSURE, {{"exposure", render_state_sanitized.exposure_}},
                      global_params);
    basic.SetOperator(OperatorType::CONTRAST, {{"contrast", render_state_sanitized.contrast_}},
                      global_params);
    basic.SetOperator(OperatorType::BLACK, {{"black", render_state_sanitized.blacks_}},
                      global_params);
    basic.SetOperator(OperatorType::WHITE, {{"white", render_state_sanitized.whites_}},
                      global_params);
    basic.SetOperator(OperatorType::SHADOWS, {{"shadows", render_state_sanitized.shadows_}},
                      global_params);
    basic.SetOperator(OperatorType::HIGHLIGHTS,
                      {{"highlights", render_state_sanitized.highlights_}}, global_params);
    basic.SetOperator(OperatorType::CURVE,
                      CurveControlPointsToParams(render_state_sanitized.curve_points_),
                      global_params);

    auto& color = exec->GetStage(PipelineStageName::Color_Adjustment);
    color.SetOperator(OperatorType::SATURATION,
                      {{"saturation", render_state_sanitized.saturation_}}, global_params);
    color.EnableOperator(OperatorType::TINT, false, global_params);
    color.SetOperator(OperatorType::COLOR_WHEEL,
                      ParamsForField(AdjustmentField::ColorWheel, render_state_sanitized),
                      global_params);
    color.EnableOperator(OperatorType::COLOR_WHEEL, true, global_params);
    color.SetOperator(OperatorType::HLS,
                      ParamsForField(AdjustmentField::Hls, render_state_sanitized), global_params);
    color.EnableOperator(OperatorType::HLS, true, global_params);

    // LUT (LMT): rebind only when the path changes. The operator's SetGlobalParams now
    // derives lmt_enabled_/dirty state from the path, and PipelineMgmtService resyncs on load.
    if (render_state_sanitized.lut_path_ != last_applied_lut_path_) {
      color.SetOperator(OperatorType::LMT, {{"ocio_lmt", render_state_sanitized.lut_path_}},
                        global_params);
      last_applied_lut_path_ = render_state_sanitized.lut_path_;
    }

    auto& detail = exec->GetStage(PipelineStageName::Detail_Adjustment);
    detail.SetOperator(
        OperatorType::SHARPEN, {{"sharpen", {{"offset", render_state_sanitized.sharpen_}}}},
        global_params);
    detail.SetOperator(OperatorType::CLARITY, {{"clarity", render_state_sanitized.clarity_}},
                       global_params);
  }
}  // namespace alcedo::ui
