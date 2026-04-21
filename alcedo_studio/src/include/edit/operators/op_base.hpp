//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <cstdint>
#include <memory>
#include <string>

#include "decoders/processor/raw_color_context.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "op_kernel.hpp"
#include "edit/operators/geometry/lens_calib_runtime.hpp"
#include "type/type.hpp"
#include "utils/color_utils.hpp"

namespace alcedo {
namespace OCIO = OCIO_NAMESPACE;

enum class PipelineStageName : int {
  Image_Loading       = 0,
  Geometry_Adjustment = 1,
  To_WorkingSpace     = 2,
  Basic_Adjustment    = 3,
  Color_Adjustment    = 4,
  Detail_Adjustment   = 5,
  Output_Transform    = 6,
  Stage_Count         = 7,
  Merged_Stage        = 8  // Special stage for merged streamable stages
};

enum class OperatorType : int {
  RAW_DECODE,
  RESIZE,
  EXPOSURE,
  CONTRAST,
  WHITE,
  BLACK,
  SHADOWS,
  HIGHLIGHTS,
  CURVE,
  HLS,
  SATURATION,
  TINT,
  VIBRANCE,
  CST,
  TO_WS,
  TO_OUTPUT,
  LMT,
  ODT,
  CLARITY,
  SHARPEN,
  COLOR_WHEEL,
  ACES_TONE_MAPPING,
  AUTO_EXPOSURE,
  UNKNOWN,  // For unrecognized operator types or placeholders
  CROP_ROTATE,
  LENS_CALIBRATION,
  COLOR_TEMP
};

enum class ColorTempMode : int {
  AS_SHOT = 0,
  CUSTOM  = 1,
};

enum class RawDecodeInputSpace : int {
  AP0    = 0,
  CAMERA = 1,
};

struct OperatorParams {
  // Basic adjustment parameters
  bool                         exposure_enabled_         = true;
  float                        exposure_offset_          = 0.0f;

  bool                         contrast_enabled_         = true;
  float                        contrast_scale_           = 0.0f;

  bool                         shadows_operator_present_ = false;
  bool                         highlights_operator_present_ = false;

  // Host-side slider state used to rebuild the GPU-only shared tone curve.
  float                        shadows_slider_value_     = 0.0f;
  float                        highlights_slider_value_  = 0.0f;

  // Shadows adjustment parameter
  bool                         shadows_enabled_          = true;
  float                        shadows_offset_           = 0.0f;
  float                        shadows_x0_               = 0.0f;
  float                        shadows_x1_               = 0.25f;
  float                        shadows_y0_               = 0.0f;
  float                        shadows_y1_               = 0.25f;
  float                        shadows_m0_               = 0.0f;
  float                        shadows_m1_               = 1.0f;
  float                        shadows_dx_               = 0.25f;

  // Highlights adjustment parameter
  bool                         highlights_enabled_       = true;
  const float                  highlights_k_             = 0.2f;
  float                        highlights_offset_        = 0.0f;
  const float                  highlights_slope_range_   = 0.8f;
  float                        highlights_m0_            = 1.0f;
  float                        highlights_m1_            = 1.0f;
  float                        highlights_x0_            = 0.2f;
  float                        highlights_y0_            = 0.2f;
  float                        highlights_y1_            = 1.0f;
  float                        highlights_dx_            = 0.8f;

  // GPU-only shared shadows/highlights tone curve parameters.
  static constexpr int         kSharedToneCurveControlPointCount = 5;
  bool                         shared_tone_curve_enabled_ = false;
  bool                         shared_tone_curve_apply_in_shadows_ = false;
  bool                         shared_tone_curve_apply_in_highlights_ = false;
  int                          shared_tone_curve_ctrl_pts_size_ = 0;
  float                        shared_tone_curve_ctrl_pts_x_[kSharedToneCurveControlPointCount] = {};
  float                        shared_tone_curve_ctrl_pts_y_[kSharedToneCurveControlPointCount] = {};
  float                        shared_tone_curve_h_[kSharedToneCurveControlPointCount - 1] = {};
  float                        shared_tone_curve_m_[kSharedToneCurveControlPointCount] = {};

  // White and Black point adjustment parameters
  bool                         white_enabled_            = true;
  float                        white_point_              = 1.0f;

  bool                         black_enabled_            = true;
  float                        black_point_              = 0.0f;

  float                        slope_                    = 1.0f;
  // HLS adjustment parameters
  static constexpr int         kHlsProfileCount          = 8;
  bool                         hls_enabled_              = true;
  float                        target_hls_[3]            = {0.0f, 0.5f, 1.0f};
  float                        hls_adjustment_[3]        = {0.0f, 0.0f, 0.0f};
  float                        hue_range_                = 15.0f;
  float                        lightness_range_          = 0.1f;
  float                        saturation_range_         = 0.1f;
  int                          hls_profile_count_        = kHlsProfileCount;
  float                        hls_profile_hues_[kHlsProfileCount] = {
      0.0f, 45.0f, 90.0f, 135.0f, 180.0f, 225.0f, 270.0f, 315.0f};
  float                        hls_profile_adjustments_[kHlsProfileCount][3] = {};
  float                        hls_profile_hue_ranges_[kHlsProfileCount] = {
      15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f};

  // Saturation adjustment parameter
  bool                         saturation_enabled_       = true;
  float                        saturation_offset_        = 0.0f;

  // Tint adjustment parameter
  bool                         tint_enabled_             = true;
  float                        tint_offset_              = 0.0f;

  // Vibrance adjustment parameter
  bool                         vibrance_enabled_         = true;
  float                        vibrance_offset_          = 0.0f;

  // Working space
  bool                         to_ws_enabled_            = true;
  bool                         is_working_space_         = true;
  bool                         to_ws_dirty_              = false;
  OCIO::ConstCPUProcessorRcPtr cpu_to_working_processor_ = nullptr;
  OCIO::ConstGPUProcessorRcPtr gpu_to_working_processor_ = nullptr;
  OCIO::BakerRcPtr             to_ws_lut_baker_          = nullptr;

  // Dynamic RAW color temperature / white balance state.
  bool                         color_temp_enabled_        = true;
  ColorTempMode                color_temp_mode_           = ColorTempMode::AS_SHOT;
  float                        color_temp_custom_cct_     = 6500.0f;
  float                        color_temp_custom_tint_    = 0.0f;
  float                        color_temp_resolved_cct_   = 6500.0f;
  float                        color_temp_resolved_tint_  = 0.0f;
  float                        color_temp_resolved_xy_[2] = {0.3127f, 0.3290f};

  bool                         raw_runtime_valid_         = false;
  RawDecodeInputSpace          raw_decode_input_space_    = RawDecodeInputSpace::AP0;
  float                        raw_cam_mul_[3]            = {1.0f, 1.0f, 1.0f};
  float                        raw_pre_mul_[3]            = {1.0f, 1.0f, 1.0f};
  float                        raw_cam_xyz_[9]            = {};
  float                        raw_rgb_cam_[9]            = {};
  std::string                  raw_camera_make_           = {};
  std::string                  raw_camera_model_          = {};
  bool                         raw_color_matrices_valid_  = false;
  double                       raw_color_matrix_1_[9]     = {};
  double                       raw_color_matrix_2_[9]     = {};
  bool                         raw_as_shot_neutral_valid_ = false;
  double                       raw_as_shot_neutral_[3]    = {};
  bool                         raw_calibration_illuminants_valid_ = false;
  double                       raw_color_matrix_1_cct_    = 2856.0;
  double                       raw_color_matrix_2_cct_    = 6504.0;
  bool                         raw_lens_metadata_valid_   = false;
  std::string                  raw_lens_make_             = {};
  std::string                  raw_lens_model_            = {};
  float                        raw_lens_focal_mm_         = 0.0f;
  float                        raw_lens_aperture_f_       = 0.0f;
  float                        raw_lens_focus_distance_m_ = 0.0f;
  float                        raw_lens_focal_35mm_       = 0.0f;
  float                        raw_lens_crop_factor_hint_ = 0.0f;

  bool                         lens_calib_enabled_          = true;
  bool                         lens_calib_runtime_dirty_    = true;
  bool                         lens_calib_runtime_valid_    = false;
  bool                         lens_calib_runtime_failed_   = false;
  bool                         lens_calib_cache_key_valid_  = false;
  uint64_t                     lens_calib_cache_key_        = 0;
  LensCalibGpuParams           lens_calib_runtime_params_   = {};

  bool                         color_temp_runtime_dirty_  = true;
  bool                         color_temp_matrices_valid_ = false;
  bool                         color_temp_cache_key_valid_ = false;
  uint64_t                     color_temp_cache_key_       = 0;
  float                        color_temp_cam_to_xyz_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float                        color_temp_cam_to_xyz_d50_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float                        color_temp_xyz_d50_to_ap1_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float                        color_temp_cam_to_ap1_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};

  // Look modification transform
  bool                         lmt_enabled_              = false;
  bool                         to_lmt_dirty_             = false;
  OCIO::ConstCPUProcessorRcPtr cpu_lmt_processor_        = nullptr;
  OCIO::ConstGPUProcessorRcPtr gpu_lmt_processor_        = nullptr;
  std::filesystem::path        lmt_lut_path_             = {};

  // Output transform runtime (ACES 2.0 or OpenDRT)
  bool                         to_output_enabled_        = true;
  bool                         to_output_dirty_          = false;
  OCIO::ConstCPUProcessorRcPtr cpu_to_output_processor_  = nullptr;
  // [UNUSED] Current approach does not use OCIO or LUT in GPU path
  OCIO::ConstGPUProcessorRcPtr gpu_to_output_processor_  = nullptr;
  OCIO::BakerRcPtr             to_output_lut_baker_      = nullptr;
  // [WIP] Ported from Academy Color Encoding System Core Transforms
  // A CUDA implementation of the CTL operators
  // https://github.com/aces-aswf/aces-core
  ColorUtils::TO_OUTPUT_Params to_output_params_         = {};

  // Curve adjustment parameters
  bool                         curve_enabled_            = true;
  std::vector<cv::Point2f>     curve_ctrl_pts_           = {};
  std::vector<float>           curve_h_                  = {};
  std::vector<float>           curve_m_                  = {};

  static constexpr int         kDetailMaxGaussianTapCount = 24;

  // Clarity adjustment parameter
  bool                         clarity_enabled_          = true;
  float                        clarity_offset_           = 0.0f;
  float                        clarity_radius_           = 5.0f;
  int                          clarity_gaussian_tap_count_ = 0;
  float                        clarity_gaussian_weights_[kDetailMaxGaussianTapCount] = {};

  // Sharpen adjustment parameter
  bool                         sharpen_enabled_          = true;
  float                        sharpen_offset_           = 0.0f;
  float                        sharpen_radius_           = 3.0f;
  float                        sharpen_threshold_        = 0.0f;
  int                          sharpen_gaussian_tap_count_ = 0;
  float                        sharpen_gaussian_weights_[kDetailMaxGaussianTapCount] = {};

  // Color wheel adjustment parameters
  bool                         color_wheel_enabled_      = true;
  float                        lift_color_offset_[3]     = {0.0f, 0.0f, 0.0f};
  float                        lift_luminance_offset_    = 0.0f;
  float                        gamma_color_offset_[3]    = {1.0f, 1.0f, 1.0f};
  float                        gamma_luminance_offset_   = 0.0f;
  float                        gain_color_offset_[3]     = {1.0f, 1.0f, 1.0f};
  float                        gain_luminance_offset_    = 0.0f;
  /// Populate the raw metadata fields from a RawRuntimeColorContext.
  /// Call this at import/load time so that downstream operators (ColorTemp, LensCalib)
  /// can resolve eagerly in their SetGlobalParams without waiting for pipeline execution.
  void PopulateRawMetadata(const RawRuntimeColorContext& ctx) {
    raw_runtime_valid_         = ctx.valid_;
    raw_decode_input_space_    = ctx.output_in_camera_space_
                                     ? RawDecodeInputSpace::CAMERA
                                     : RawDecodeInputSpace::AP0;
    for (int i = 0; i < 3; ++i) {
      raw_cam_mul_[i] = ctx.cam_mul_[i];
      raw_pre_mul_[i] = ctx.pre_mul_[i];
    }
    for (int i = 0; i < 9; ++i) {
      raw_cam_xyz_[i] = ctx.cam_xyz_[i];
      raw_rgb_cam_[i] = ctx.rgb_cam_[i];
    }
    raw_camera_make_           = ctx.camera_make_;
    raw_camera_model_          = ctx.camera_model_;
    raw_color_matrices_valid_  = ctx.color_matrices_valid_;
    for (int i = 0; i < 9; ++i) {
      raw_color_matrix_1_[i] = ctx.color_matrix_1_[i];
      raw_color_matrix_2_[i] = ctx.color_matrix_2_[i];
    }
    raw_as_shot_neutral_valid_ = ctx.as_shot_neutral_valid_;
    for (int i = 0; i < 3; ++i) {
      raw_as_shot_neutral_[i] = ctx.as_shot_neutral_[i];
    }
    raw_calibration_illuminants_valid_ = ctx.calibration_illuminants_valid_;
    raw_color_matrix_1_cct_            = ctx.color_matrix_1_cct_;
    raw_color_matrix_2_cct_            = ctx.color_matrix_2_cct_;
    raw_lens_metadata_valid_   = ctx.lens_metadata_valid_;
    raw_lens_make_             = ctx.lens_make_;
    raw_lens_model_            = ctx.lens_model_;
    raw_lens_focal_mm_         = ctx.focal_length_mm_;
    raw_lens_aperture_f_       = ctx.aperture_f_number_;
    raw_lens_focus_distance_m_ = ctx.focus_distance_m_;
    raw_lens_focal_35mm_       = ctx.focal_35mm_mm_;
    raw_lens_crop_factor_hint_ = ctx.crop_factor_hint_;

    // Mark dependent operators dirty so they re-resolve on next SetGlobalParams call
    lens_calib_runtime_dirty_  = true;
    color_temp_runtime_dirty_  = true;
  }};

class IOperatorBase {
 public:
  /**
   * @brief Apply the adjustment from the operator
   *
   * @param input
   * @return ImageBuffer
   */
  virtual void Apply(std::shared_ptr<ImageBuffer> input)     = 0;

  /**
   * @brief Only used in Resize operator. Other operators should throw exception.
   * 
   * @param input 
   */
  virtual void ApplyGPU(std::shared_ptr<ImageBuffer> input) = 0;
  /**
   * @brief Set the parameters of this operator from JSON
   *
   * @param params
   */
  virtual auto GetParams() const -> nlohmann::json           = 0;
  /**
   * @brief Get JSON parameter for this operator
   *
   * @return nlohmann::json
   */
  virtual void SetParams(const nlohmann::json&)              = 0;

  virtual void SetGlobalParams(OperatorParams& params) const = 0;

  virtual void EnableGlobalParams(OperatorParams& params, bool enable)               = 0;

  virtual auto GetScriptName() const -> std::string          = 0;

  virtual auto GetPriorityLevel() const -> PriorityLevel     = 0;

  virtual auto GetStage() const -> PipelineStageName         = 0;

  virtual auto GetOperatorType() const -> OperatorType       = 0;

  // virtual auto ToKernel_Vec() const -> Kernel            = 0;

  virtual ~IOperatorBase()                                   = default;
};
/**
 * @brief A base class for all operators
 *
 * @tparam Derived CRTP derived class
 */
template <typename Derived>
class OperatorBase : public IOperatorBase {
 public:
  /**
   * @brief Get the canonical name of the operator (for display)
   *
   * @return std::string
   */
  virtual auto GetCanonicalName() const -> std::string {
    return std::string(Derived::canonical_name_);
  }
  /**
   * @brief Get the script name of the operator (for JSON serialization)
   *
   * @return std::string
   */
  auto GetScriptName() const -> std::string override { return std::string(Derived::script_name_); }

  auto GetPriorityLevel() const -> PriorityLevel override { return Derived::priority_level_; }

  auto GetStage() const -> PipelineStageName override { return Derived::affiliation_stage_; }

  auto GetOperatorType() const -> OperatorType override { return Derived::operator_type_; }
};

struct OpStream {
  std::vector<std::shared_ptr<IOperatorBase>> ops_;

  bool AddToStream(const std::shared_ptr<IOperatorBase>& op) {
    // Ensure all kernels in the stream are of the same type
    ops_.push_back(op);
    return true;
  }

  void Clear() { ops_.clear(); }
};

};  // namespace alcedo
