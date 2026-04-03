//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <array>
#include <cuda_runtime.h>
#include <driver_types.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "edit/operators/op_base.hpp"
#include "edit/operators/GPU_kernels/fused_param.hpp"
#include "utils/lut/cube_lut.hpp"

#define GPU_FUNC __device__ __forceinline__

namespace puerhlab {
struct GPU_LUT3D {
  cudaArray_t         array_              = nullptr;
  cudaTextureObject_t texture_object_     = 0;
  int                 edge_size_          = 0;
  bool                borrowed_           = false;

  GPU_LUT3D()                             = default;

  GPU_LUT3D(const GPU_LUT3D&)             = default;

  GPU_LUT3D& operator=(const GPU_LUT3D&)  = default;

  GPU_LUT3D(GPU_LUT3D&& other)            = default;

  GPU_LUT3D& operator=(GPU_LUT3D&& other) = default;

  void       Reset() {
    if (borrowed_) {
      array_          = nullptr;
      texture_object_ = 0;
      edge_size_      = 0;
      borrowed_       = false;
      return;
    }
    if (texture_object_) {
      cudaDestroyTextureObject(texture_object_);
      texture_object_ = 0;
    }
    if (array_) {
      cudaFreeArray(array_);
      array_ = nullptr;
    }
    edge_size_ = 0;
  }
};

struct GPU_JMhParams {
  float MATRIX_RGB_to_CAM16_c_[9];
  float MATRIX_CAM16_c_to_RGB_[9];
  float MATRIX_cone_response_to_Aab_[9];
  float MATRIX_Aab_to_cone_response_[9];
  float F_L_n_;  // F_L normalized
  float cz_;
  float inv_cz_;  // 1/cz
  float A_w_J_;
  float inv_A_w_J_;  // 1/A_w_J
};

struct GPU_TSParams {
  float n_;
  float n_r_;
  float g_;
  float t_1_;
  float c_t_;
  float s_2_;
  float u_2_;
  float m_2_;
  float forward_limit_;
  float inverse_limit_;
  float log_peak_;
};

template <typename T>
struct GPU_Table1D {
  cudaTextureObject_t texture_object_ = 0;
  void*               dev_ptr_        = nullptr;
  size_t              count_          = 0;

  GPU_Table1D()                       = default;

  void Reset() {
    if (texture_object_) {
      cudaDestroyTextureObject(texture_object_);
      texture_object_ = 0;
    }
    if (dev_ptr_) {
      cudaFree(dev_ptr_);
      dev_ptr_ = nullptr;
    }
    count_ = 0;
  }
};

// Note: This is a host-side helper that creates a texture object bound to linear
// device memory (cudaResourceTypeLinear). It does not use hardware filtering.
// We intentionally keep the CTL lookup / interpolation logic explicit in CUDA
// code for correctness and maintainability.
template <typename T>
static GPU_Table1D<T> Create1DLinearTableTextureObject(const T* host_data, size_t count) {
  GPU_Table1D<T> table;
  if (!host_data || count == 0) {
    return table;
  }

  table.count_       = count;

  const size_t bytes = sizeof(T) * count;
  cudaMalloc(&table.dev_ptr_, bytes);
  cudaMemcpy(table.dev_ptr_, host_data, bytes, cudaMemcpyHostToDevice);

  cudaResourceDesc res_desc       = {};
  res_desc.resType                = cudaResourceTypeLinear;
  res_desc.res.linear.devPtr      = table.dev_ptr_;
  res_desc.res.linear.desc        = cudaCreateChannelDesc<T>();
  res_desc.res.linear.sizeInBytes = bytes;

  cudaTextureDesc tex_desc        = {};
  tex_desc.normalizedCoords       = 0;
  tex_desc.filterMode             = cudaFilterModePoint;
  tex_desc.readMode               = cudaReadModeElementType;
  tex_desc.addressMode[0]         = cudaAddressModeClamp;

  cudaCreateTextureObject(&table.texture_object_, &res_desc, &tex_desc, nullptr);
  return table;
}

struct GPU_ODTParams {
  float               peak_luminance_ = 100.0f;

  // JMh parameters
  GPU_JMhParams       input_params_;
  GPU_JMhParams       reach_params_;
  GPU_JMhParams       limit_params_;

  // Tonescale parameters
  GPU_TSParams        ts_;

  // Shared compression parameters
  float               limit_J_max;
  float               model_gamma_inv;
  GPU_Table1D<float>  table_reach_M_;
  std::uintptr_t      host_table_reach_M_id_ = 0;

  // Chroma compression parameters
  float               sat;
  float               sat_thr;
  float               compr;
  float               chroma_compress_scale;

  // Gamut compression parameters
  float               mid_J;
  float               focus_dist;
  float               lower_hull_gamma_inv;
  GPU_Table1D<float>  table_hues_;
  std::uintptr_t      host_table_hues_id_ = 0;

  // Packed as float4{J, M, h}
  GPU_Table1D<float4> table_gamut_cusps_;
  std::uintptr_t      host_table_gamut_cusps_id_ = 0;

  GPU_Table1D<float>  table_upper_hull_gamma_;
  std::uintptr_t      host_table_upper_hull_gamma_id_ = 0;

  int                 hue_linearity_search_range[2]   = {0, 1};

  void                Reset() {
    table_reach_M_.Reset();
    table_hues_.Reset();
    table_gamut_cusps_.Reset();
    table_upper_hull_gamma_.Reset();
    host_table_reach_M_id_          = 0;
    host_table_hues_id_             = 0;
    host_table_gamut_cusps_id_      = 0;
    host_table_upper_hull_gamma_id_ = 0;
    hue_linearity_search_range[0]   = 0;
    hue_linearity_search_range[1]   = 1;
  }
};

struct GPU_OpenDRTParams {
  int   tn_hcon_enable_ = 0;
  int   tn_lcon_enable_ = 0;
  int   pt_enable_      = 1;
  int   ptl_enable_     = 1;
  int   ptm_enable_     = 1;
  int   brl_enable_     = 1;
  int   brlp_enable_    = 1;
  int   hc_enable_      = 1;
  int   hs_rgb_enable_  = 1;
  int   hs_cmy_enable_  = 1;
  int   creative_white_ = 2;
  int   surround_       = 2;
  int   clamp_          = 1;
  int   display_gamut_  = 0;
  int   display_eotf_   = 1;

  float tn_con_         = 1.66f;
  float tn_sh_          = 0.5f;
  float tn_toe_         = 0.003f;
  float tn_off_         = 0.005f;
  float tn_hcon_        = 0.0f;
  float tn_hcon_pv_     = 1.0f;
  float tn_hcon_st_     = 4.0f;
  float tn_lcon_        = 0.0f;
  float tn_lcon_w_      = 0.5f;
  float cwp_lm_         = 0.25f;
  float rs_sa_          = 0.35f;
  float rs_rw_          = 0.25f;
  float rs_bw_          = 0.55f;
  float pt_lml_         = 0.25f;
  float pt_lml_r_       = 0.5f;
  float pt_lml_g_       = 0.0f;
  float pt_lml_b_       = 0.1f;
  float pt_lmh_         = 0.25f;
  float pt_lmh_r_       = 0.5f;
  float pt_lmh_b_       = 0.0f;
  float ptl_c_          = 0.06f;
  float ptl_m_          = 0.08f;
  float ptl_y_          = 0.06f;
  float ptm_low_        = 0.4f;
  float ptm_low_rng_    = 0.25f;
  float ptm_low_st_     = 0.5f;
  float ptm_high_       = -0.8f;
  float ptm_high_rng_   = 0.35f;
  float ptm_high_st_    = 0.4f;
  float brl_            = 0.0f;
  float brl_r_          = -2.5f;
  float brl_g_          = -1.5f;
  float brl_b_          = -1.5f;
  float brl_rng_        = 0.5f;
  float brl_st_         = 0.35f;
  float brlp_           = -0.5f;
  float brlp_r_         = -1.25f;
  float brlp_g_         = -1.25f;
  float brlp_b_         = -0.25f;
  float hc_r_           = 1.0f;
  float hc_r_rng_       = 0.3f;
  float hs_r_           = 0.6f;
  float hs_r_rng_       = 0.6f;
  float hs_g_           = 0.35f;
  float hs_g_rng_       = 1.0f;
  float hs_b_           = 0.66f;
  float hs_b_rng_       = 1.0f;
  float hs_c_           = 0.25f;
  float hs_c_rng_       = 1.0f;
  float hs_m_           = 0.0f;
  float hs_m_rng_       = 1.0f;
  float hs_y_           = 0.0f;
  float hs_y_rng_       = 1.0f;

  float ts_x1_          = 0.0f;
  float ts_y1_          = 0.0f;
  float ts_x0_          = 0.0f;
  float ts_y0_          = 0.0f;
  float ts_s0_          = 0.0f;
  float ts_p_           = 0.0f;
  float ts_s10_         = 0.0f;
  float ts_m1_          = 0.0f;
  float ts_m2_          = 0.0f;
  float ts_s_           = 0.0f;
  float ts_dsc_         = 0.0f;
  float pt_cmp_Lf_      = 0.0f;
  float s_Lp100_        = 0.0f;
  float ts_s1_          = 0.0f;
};

struct GPU_TO_OUTPUT_Params {
  GPU_ODTMethod    method_               = GPU_ODTMethod::OPEN_DRT;
  GPU_ODTParams    aces_params_          = {};
  GPU_OpenDRTParams open_drt_params_     = {};
  float            limit_to_display_matx[9];
  float            display_linear_scale_ = 1.0f;
  GPU_EOTF         eotf                  = GPU_EOTF::LINEAR;

  void Reset() {
    aces_params_.Reset();
    open_drt_params_ = {};
    display_linear_scale_ = 1.0f;
    eotf = GPU_EOTF::LINEAR;
  }
};

struct GPUOperatorParams {
  // Basic adjustment parameters
  bool                 exposure_enabled_       = true;
  float                exposure_offset_        = 0.0f;

  bool                 contrast_enabled_       = true;
  float                contrast_scale_         = 0.0f;

  // Shadows adjustment parameter
  bool                 shadows_enabled_        = true;
  float                shadows_offset_         = 0.0f;
  float                shadows_x0_             = 0.0f;
  float                shadows_x1_             = 0.25f;
  float                shadows_y0_             = 0.0f;
  float                shadows_y1_             = 0.25f;
  float                shadows_m0_             = 0.0f;
  float                shadows_m1_             = 1.0f;
  float                shadows_dx_             = 0.25f;

  // Highlights adjustment parameter
  bool                 highlights_enabled_     = true;
  float                highlights_k_           = 0.2f;
  float                highlights_offset_      = 0.0f;
  float                highlights_slope_range_ = 0.8f;
  float                highlights_m0_          = 1.0f;
  float                highlights_m1_          = 1.0f;
  float                highlights_x0_          = 0.2f;
  float                highlights_y0_          = 0.2f;
  float                highlights_y1_          = 1.0f;
  float                highlights_dx_          = 0.8f;

  bool                 shared_tone_curve_enabled_ = false;
  bool                 shared_tone_curve_apply_in_shadows_ = false;
  bool                 shared_tone_curve_apply_in_highlights_ = false;
  int                  shared_tone_curve_ctrl_pts_size_ = 0;
  float                shared_tone_curve_ctrl_pts_x_[OperatorParams::kSharedToneCurveControlPointCount] = {};
  float                shared_tone_curve_ctrl_pts_y_[OperatorParams::kSharedToneCurveControlPointCount] = {};
  float                shared_tone_curve_h_[OperatorParams::kSharedToneCurveControlPointCount - 1] = {};
  float                shared_tone_curve_m_[OperatorParams::kSharedToneCurveControlPointCount] = {};

  // White and Black point adjustment parameters
  bool                 white_enabled_          = true;
  float                white_point_            = 1.0f;

  bool                 black_enabled_          = true;
  float                black_point_            = 0.0f;

  float                slope_                  = 1.0f;
  // HLS adjustment parameters
  bool                 hls_enabled_            = true;
  float                target_hls_[3]          = {0.0f, 0.5f, 1.0f};
  float                hls_adjustment_[3]      = {0.0f, 0.0f, 0.0f};
  float                hue_range_              = 15.0f;
  float                lightness_range_        = 0.1f;
  float                saturation_range_       = 0.1f;
  int                  hls_profile_count_      = OperatorParams::kHlsProfileCount;
  float                hls_profile_hues_[OperatorParams::kHlsProfileCount] = {
      0.0f, 45.0f, 90.0f, 135.0f, 180.0f, 225.0f, 270.0f, 315.0f};
  float                hls_profile_adjustments_[OperatorParams::kHlsProfileCount][3] = {};
  float                hls_profile_hue_ranges_[OperatorParams::kHlsProfileCount] = {
      15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f};

  // Saturation adjustment parameter
  bool                 saturation_enabled_     = true;
  float                saturation_offset_      = 0.0f;

  // Tint adjustment parameter
  bool                 tint_enabled_           = true;
  float                tint_offset_            = 0.0f;

  // Vibrance adjustment parameter
  bool                 vibrance_enabled_       = true;
  float                vibrance_offset_        = 0.0f;

  // Working space
  bool                 to_ws_enabled_          = true;
  GPU_LUT3D            to_ws_lut_              = {};
  // TODO: NOT IMPLEMENTED

  // RAW color temperature runtime state.
  bool                 color_temp_enabled_      = true;
  int                  color_temp_mode_         = 0;  // 0: as_shot, 1: custom
  float                color_temp_custom_cct_   = 6500.0f;
  float                color_temp_custom_tint_  = 0.0f;
  float                color_temp_resolved_cct_ = 6500.0f;
  float                color_temp_resolved_tint_ = 0.0f;
  float                color_temp_resolved_xy_[2] = {0.3127f, 0.3290f};

  bool                 raw_runtime_valid_       = false;
  int                  raw_decode_input_space_  = 0;  // 0: AP0, 1: camera
  float                raw_cam_mul_[3]          = {1.0f, 1.0f, 1.0f};
  float                raw_pre_mul_[3]          = {1.0f, 1.0f, 1.0f};
  float                raw_cam_xyz_[9]          = {};

  bool                 color_temp_matrices_valid_ = false;
  float                color_temp_cam_to_xyz_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float                color_temp_cam_to_xyz_d50_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float                color_temp_xyz_d50_to_ap1_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float                color_temp_cam_to_ap1_[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};

  // Look modification transform
  bool                 lmt_enabled_            = false;
  GPU_LUT3D            lmt_lut_                = {};
  // TODO: NOT IMPLEMENTED

  // Output transform
  bool                 to_output_enabled_      = true;
  GPU_LUT3D            to_output_lut_          = {};
  GPU_TO_OUTPUT_Params to_output_params_       = {};

  // Curve adjustment parameters
  static constexpr int kMaxCurveControlPoints  = 32;
  bool                 curve_enabled_          = true;
  int                  curve_ctrl_pts_size_    = 0;
  float                curve_ctrl_pts_x_[kMaxCurveControlPoints] = {};
  float                curve_ctrl_pts_y_[kMaxCurveControlPoints] = {};
  float                curve_h_[kMaxCurveControlPoints - 1]      = {};
  float                curve_m_[kMaxCurveControlPoints]          = {};

  // Clarity adjustment parameter
  bool                 clarity_enabled_        = true;
  float                clarity_offset_         = 0.0f;
  float                clarity_radius_         = 5.0f;
  int                  clarity_gaussian_tap_count_ = 0;
  float                clarity_gaussian_weights_[OperatorParams::kDetailMaxGaussianTapCount] = {};

  // Sharpen adjustment parameter
  bool                 sharpen_enabled_        = true;
  float                sharpen_offset_         = 0.0f;
  float                sharpen_radius_         = 3.0f;
  float                sharpen_threshold_      = 0.0f;
  int                  sharpen_gaussian_tap_count_ = 0;
  float                sharpen_gaussian_weights_[OperatorParams::kDetailMaxGaussianTapCount] = {};

  // CDL SOP wheel terms:
  // out = clamp(pow(max(in * slope + offset, 0), power), 0, 1)
  // lift -> offset, gain -> slope, gamma -> power
  bool                 color_wheel_enabled_    = true;
  float                lift_color_offset_[3]   = {0.0f, 0.0f, 0.0f};
  float                lift_luminance_offset_  = 0.0f;
  float                gamma_color_offset_[3]  = {1.0f, 1.0f, 1.0f};
  float                gamma_luminance_offset_ = 0.0f;
  float                gain_color_offset_[3]   = {1.0f, 1.0f, 1.0f};
  float                gain_luminance_offset_  = 0.0f;
};

struct CudaFusedResources {
  FusedOperatorParams common_params_ = {};
  GPUOperatorParams   uploaded_params_ = {};

  void Reset() {
    uploaded_params_.to_ws_lut_.Reset();
    uploaded_params_.lmt_lut_.Reset();
    uploaded_params_.to_output_lut_.Reset();
    uploaded_params_.to_output_params_.Reset();
  }
};

class CudaFusedParamUploader {
 public:
  static auto Upload(const FusedOperatorParams& fused_params, OperatorParams& cpu_params,
                     CudaFusedResources& orig_resources) -> CudaFusedResources {
    CudaFusedResources resources = orig_resources;
    resources.common_params_     = fused_params;
    GPUOperatorParams& gpu_params = resources.uploaded_params_;

    gpu_params.exposure_enabled_       = fused_params.exposure_enabled_;
    gpu_params.exposure_offset_        = fused_params.exposure_offset_;

    gpu_params.contrast_enabled_       = fused_params.contrast_enabled_;
    gpu_params.contrast_scale_         = fused_params.contrast_scale_;

    gpu_params.shadows_enabled_        = fused_params.shadows_enabled_;
    gpu_params.shadows_offset_         = fused_params.shadows_offset_;
    gpu_params.shadows_x0_             = fused_params.shadows_x0_;
    gpu_params.shadows_x1_             = fused_params.shadows_x1_;
    gpu_params.shadows_y0_             = fused_params.shadows_y0_;
    gpu_params.shadows_y1_             = fused_params.shadows_y1_;
    gpu_params.shadows_m0_             = fused_params.shadows_m0_;
    gpu_params.shadows_m1_             = fused_params.shadows_m1_;
    gpu_params.shadows_dx_             = fused_params.shadows_dx_;

    gpu_params.highlights_enabled_     = fused_params.highlights_enabled_;
    gpu_params.highlights_k_           = fused_params.highlights_k_;
    gpu_params.highlights_offset_      = fused_params.highlights_offset_;
    gpu_params.highlights_slope_range_ = fused_params.highlights_slope_range_;
    gpu_params.highlights_m0_          = fused_params.highlights_m0_;
    gpu_params.highlights_m1_          = fused_params.highlights_m1_;
    gpu_params.highlights_x0_          = fused_params.highlights_x0_;
    gpu_params.highlights_y0_          = fused_params.highlights_y0_;
    gpu_params.highlights_y1_          = fused_params.highlights_y1_;
    gpu_params.highlights_dx_          = fused_params.highlights_dx_;
    gpu_params.shared_tone_curve_enabled_ = fused_params.shared_tone_curve_enabled_;
    gpu_params.shared_tone_curve_apply_in_shadows_ =
        fused_params.shared_tone_curve_apply_in_shadows_;
    gpu_params.shared_tone_curve_apply_in_highlights_ =
        fused_params.shared_tone_curve_apply_in_highlights_;
    gpu_params.shared_tone_curve_ctrl_pts_size_ = fused_params.shared_tone_curve_ctrl_pts_size_;
    for (int i = 0; i < OperatorParams::kSharedToneCurveControlPointCount; ++i) {
      gpu_params.shared_tone_curve_ctrl_pts_x_[i] = fused_params.shared_tone_curve_ctrl_pts_x_[i];
      gpu_params.shared_tone_curve_ctrl_pts_y_[i] = fused_params.shared_tone_curve_ctrl_pts_y_[i];
      gpu_params.shared_tone_curve_m_[i]          = fused_params.shared_tone_curve_m_[i];
      if (i < OperatorParams::kSharedToneCurveControlPointCount - 1) {
        gpu_params.shared_tone_curve_h_[i] = fused_params.shared_tone_curve_h_[i];
      }
    }

    gpu_params.white_enabled_          = fused_params.white_enabled_;
    gpu_params.white_point_            = fused_params.white_point_;

    gpu_params.black_enabled_          = fused_params.black_enabled_;
    gpu_params.black_point_            = fused_params.black_point_;

    gpu_params.slope_                  = fused_params.slope_;

    gpu_params.hls_enabled_            = fused_params.hls_enabled_;
    for (int i = 0; i < 3; ++i) {
      gpu_params.target_hls_[i]     = fused_params.target_hls_[i];
      gpu_params.hls_adjustment_[i] = fused_params.hls_adjustment_[i];
    }
    gpu_params.hue_range_          = fused_params.hue_range_;
    gpu_params.lightness_range_    = fused_params.lightness_range_;
    gpu_params.saturation_range_   = fused_params.saturation_range_;
    gpu_params.hls_profile_count_  = fused_params.hls_profile_count_;
    for (int i = 0; i < OperatorParams::kHlsProfileCount; ++i) {
      gpu_params.hls_profile_hues_[i]            = fused_params.hls_profile_hues_[i];
      gpu_params.hls_profile_hue_ranges_[i]      = fused_params.hls_profile_hue_ranges_[i];
      gpu_params.hls_profile_adjustments_[i][0]  = fused_params.hls_profile_adjustments_[i][0];
      gpu_params.hls_profile_adjustments_[i][1]  = fused_params.hls_profile_adjustments_[i][1];
      gpu_params.hls_profile_adjustments_[i][2]  = fused_params.hls_profile_adjustments_[i][2];
    }
    gpu_params.saturation_enabled_ = fused_params.saturation_enabled_;
    gpu_params.saturation_offset_  = fused_params.saturation_offset_;
    gpu_params.tint_enabled_       = fused_params.tint_enabled_;
    gpu_params.tint_offset_        = fused_params.tint_offset_;
    gpu_params.vibrance_enabled_   = fused_params.vibrance_enabled_;
    gpu_params.vibrance_offset_    = fused_params.vibrance_offset_;
    gpu_params.to_ws_enabled_      = fused_params.to_ws_enabled_;
    // if (cpu_params.to_ws_dirty) {
    //   gpu_params.to_ws_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_ws_lut        = CreateLUTTextureObject(cpu_params.to_ws_lut_baker);
    //   cpu_params.to_ws_dirty      = false;
    // } else {
    //   gpu_params.to_ws_lut = orig_resources.uploaded_params_.to_ws_lut;
    // }

    gpu_params.color_temp_enabled_       = fused_params.color_temp_enabled_;
    gpu_params.color_temp_mode_          = fused_params.color_temp_mode_;
    gpu_params.color_temp_custom_cct_    = fused_params.color_temp_custom_cct_;
    gpu_params.color_temp_custom_tint_   = fused_params.color_temp_custom_tint_;
    gpu_params.color_temp_resolved_cct_  = fused_params.color_temp_resolved_cct_;
    gpu_params.color_temp_resolved_tint_ = fused_params.color_temp_resolved_tint_;
    gpu_params.color_temp_resolved_xy_[0] = fused_params.color_temp_resolved_xy_[0];
    gpu_params.color_temp_resolved_xy_[1] = fused_params.color_temp_resolved_xy_[1];

    gpu_params.raw_runtime_valid_       = fused_params.raw_runtime_valid_;
    gpu_params.raw_decode_input_space_  = fused_params.raw_decode_input_space_;
    for (int i = 0; i < 3; ++i) {
      gpu_params.raw_cam_mul_[i] = fused_params.raw_cam_mul_[i];
      gpu_params.raw_pre_mul_[i] = fused_params.raw_pre_mul_[i];
    }
    for (int i = 0; i < 9; ++i) {
      gpu_params.raw_cam_xyz_[i]              = fused_params.raw_cam_xyz_[i];
      gpu_params.color_temp_cam_to_xyz_[i]    = fused_params.color_temp_cam_to_xyz_[i];
      gpu_params.color_temp_cam_to_xyz_d50_[i] = fused_params.color_temp_cam_to_xyz_d50_[i];
      gpu_params.color_temp_xyz_d50_to_ap1_[i] = fused_params.color_temp_xyz_d50_to_ap1_[i];
      gpu_params.color_temp_cam_to_ap1_[i]     = fused_params.color_temp_cam_to_ap1_[i];
    }
    gpu_params.color_temp_matrices_valid_ = fused_params.color_temp_matrices_valid_;

    gpu_params.lmt_enabled_        = fused_params.lmt_enabled_;
    const bool lmt_gpu_lut_missing =
        (orig_resources.uploaded_params_.lmt_lut_.texture_object_ == 0 ||
         orig_resources.uploaded_params_.lmt_lut_.edge_size_ <= 1);
    if (cpu_params.lmt_enabled_ && (cpu_params.to_lmt_dirty_ || lmt_gpu_lut_missing)) {
      if (cpu_params.lmt_lut_path_.empty()) {
        throw std::runtime_error(
            "GPUParamsConverter: LMT is enabled but lmt_lut_path_ is empty.");
      }
      gpu_params.lmt_lut_.Reset();  // Explicitly reset existing LUT
      gpu_params.lmt_lut_      = CreateLUTTextureObject(cpu_params.lmt_lut_path_);
      cpu_params.to_lmt_dirty_ = false;
    } else if (!cpu_params.lmt_enabled_) {
      gpu_params.lmt_lut_.Reset();
    } else {
      gpu_params.lmt_lut_ = orig_resources.uploaded_params_.lmt_lut_;
    }

    gpu_params.to_output_enabled_   = fused_params.to_output_enabled_;
    // if (cpu_params.to_output_dirty) {
    //   gpu_params.to_output_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_output_lut        = CreateLUTTextureObject(cpu_params.to_output_lut_baker);
    //   cpu_params.to_output_dirty      = false;
    // } else {
    //   gpu_params.to_output_lut = orig_resources.uploaded_params_.to_output_lut;
    // }

    gpu_params.curve_enabled_       = fused_params.curve_enabled_;
    gpu_params.curve_ctrl_pts_size_ = fused_params.curve_ctrl_pts_size_;
    for (int i = 0; i < GPUOperatorParams::kMaxCurveControlPoints; ++i) {
      gpu_params.curve_ctrl_pts_x_[i] = fused_params.curve_ctrl_pts_x_[i];
      gpu_params.curve_ctrl_pts_y_[i] = fused_params.curve_ctrl_pts_y_[i];
      gpu_params.curve_m_[i]          = fused_params.curve_m_[i];
      if (i < GPUOperatorParams::kMaxCurveControlPoints - 1) {
        gpu_params.curve_h_[i] = fused_params.curve_h_[i];
      }
    }
    gpu_params.clarity_enabled_     = fused_params.clarity_enabled_;
    gpu_params.clarity_offset_      = fused_params.clarity_offset_;
    gpu_params.clarity_radius_      = fused_params.clarity_radius_;
    gpu_params.clarity_gaussian_tap_count_ =
        std::clamp(fused_params.clarity_gaussian_tap_count_, 0,
                   OperatorParams::kDetailMaxGaussianTapCount);
    for (int i = 0; i < OperatorParams::kDetailMaxGaussianTapCount; ++i) {
      gpu_params.clarity_gaussian_weights_[i] = fused_params.clarity_gaussian_weights_[i];
    }
    gpu_params.sharpen_enabled_     = fused_params.sharpen_enabled_;
    gpu_params.sharpen_offset_      = fused_params.sharpen_offset_;
    gpu_params.sharpen_radius_      = fused_params.sharpen_radius_;
    gpu_params.sharpen_threshold_   = fused_params.sharpen_threshold_;
    gpu_params.sharpen_gaussian_tap_count_ =
        std::clamp(fused_params.sharpen_gaussian_tap_count_, 0,
                   OperatorParams::kDetailMaxGaussianTapCount);
    for (int i = 0; i < OperatorParams::kDetailMaxGaussianTapCount; ++i) {
      gpu_params.sharpen_gaussian_weights_[i] = fused_params.sharpen_gaussian_weights_[i];
    }
    gpu_params.color_wheel_enabled_ = fused_params.color_wheel_enabled_;
    for (int i = 0; i < 3; ++i) {
      gpu_params.lift_color_offset_[i]  = fused_params.lift_color_offset_[i];
      gpu_params.gamma_color_offset_[i] = fused_params.gamma_color_offset_[i];
      gpu_params.gain_color_offset_[i]  = fused_params.gain_color_offset_[i];
    }
    gpu_params.lift_luminance_offset_  = fused_params.lift_luminance_offset_;
    gpu_params.gamma_luminance_offset_ = fused_params.gamma_luminance_offset_;
    gpu_params.gain_luminance_offset_  = fused_params.gain_luminance_offset_;

    // ----------------------------------------------------------------------
    // Generic ODT runtime upload. ACES keeps its precomputed table upload path,
    // while OpenDRT copies only the resolved scalar runtime.
    // ----------------------------------------------------------------------
    if (cpu_params.to_output_dirty_) {
      auto&       to_output_gpu = gpu_params.to_output_params_;
      const auto& to_output_cpu = cpu_params.to_output_params_;

      auto copy33 = [](const cv::Matx33f& m, float out[9]) {
        out[0] = m(0, 0);
        out[1] = m(0, 1);
        out[2] = m(0, 2);
        out[3] = m(1, 0);
        out[4] = m(1, 1);
        out[5] = m(1, 2);
        out[6] = m(2, 0);
        out[7] = m(2, 1);
        out[8] = m(2, 2);
      };

      copy33(to_output_cpu.limit_to_display_matx_, to_output_gpu.limit_to_display_matx);
      to_output_gpu.eotf                 =
          static_cast<GPU_EOTF>(static_cast<int>(to_output_cpu.eotf_));
      to_output_gpu.method_              =
          static_cast<GPU_ODTMethod>(static_cast<int>(to_output_cpu.method_));
      to_output_gpu.display_linear_scale_ = to_output_cpu.display_linear_scale_;

      if (!cpu_params.to_output_enabled_) {
        to_output_gpu.Reset();
        copy33(to_output_cpu.limit_to_display_matx_, to_output_gpu.limit_to_display_matx);
        to_output_gpu.eotf                 =
            static_cast<GPU_EOTF>(static_cast<int>(to_output_cpu.eotf_));
        to_output_gpu.method_              =
            static_cast<GPU_ODTMethod>(static_cast<int>(to_output_cpu.method_));
        to_output_gpu.display_linear_scale_ = to_output_cpu.display_linear_scale_;
        cpu_params.to_output_dirty_         = false;
      } else if (to_output_cpu.method_ == ColorUtils::ODTMethod::ACES_2_0) {
        const auto& odt_check = to_output_cpu.aces_params_;
        const bool  missing_tables = (!odt_check.table_reach_M_) || (!odt_check.table_hues_) ||
                                    (!odt_check.table_upper_hull_gammas_) ||
                                    (!odt_check.table_gamut_cusps_);
        if (missing_tables) {
          std::ostringstream oss;
          oss << "GPUParamsConverter: ACES ODT tables are not initialized:";
          if (!odt_check.table_reach_M_) oss << " table_reach_M_";
          if (!odt_check.table_hues_) oss << " table_hues_";
          if (!odt_check.table_upper_hull_gammas_) oss << " table_upper_hull_gammas_";
          if (!odt_check.table_gamut_cusps_) oss << " table_gamut_cusps_";
          oss << ". Ensure CPU ODT precompute runs before executing the GPU pipeline.";
          throw std::runtime_error(oss.str());
        }

        auto copy_jmh = [&](const ColorUtils::JMhParams& src, GPU_JMhParams& dst) {
          copy33(src.MATRIX_RGB_to_CAM16_c_, dst.MATRIX_RGB_to_CAM16_c_);
          copy33(src.MATRIX_CAM16_c_to_RGB_, dst.MATRIX_CAM16_c_to_RGB_);
          copy33(src.MATRIX_cone_response_to_Aab_, dst.MATRIX_cone_response_to_Aab_);
          copy33(src.MATRIX_Aab_to_cone_response_, dst.MATRIX_Aab_to_cone_response_);
          dst.F_L_n_     = src.F_L_n_;
          dst.cz_        = src.cz_;
          dst.inv_cz_    = src.inv_cz_;
          dst.A_w_J_     = (src.inv_A_w_J_ != 0.f) ? (1.f / src.inv_A_w_J_) : 0.f;
          dst.inv_A_w_J_ = src.inv_A_w_J_;
        };

        const auto& odt_cpu      = to_output_cpu.aces_params_;
        auto&       odt_gpu      = to_output_gpu.aces_params_;
        to_output_gpu.open_drt_params_ = {};

        odt_gpu.peak_luminance_       = odt_cpu.peak_luminance_;
        odt_gpu.limit_J_max           = odt_cpu.limit_J_max_;
        odt_gpu.model_gamma_inv       = odt_cpu.model_gamma_inv_;
        odt_gpu.ts_.n_                = odt_cpu.ts_params_.n_;
        odt_gpu.ts_.n_r_              = odt_cpu.ts_params_.n_r_;
        odt_gpu.ts_.g_                = odt_cpu.ts_params_.g_;
        odt_gpu.ts_.t_1_              = odt_cpu.ts_params_.t_1_;
        odt_gpu.ts_.c_t_              = odt_cpu.ts_params_.c_t_;
        odt_gpu.ts_.s_2_              = odt_cpu.ts_params_.s_2_;
        odt_gpu.ts_.u_2_              = odt_cpu.ts_params_.u_2_;
        odt_gpu.ts_.m_2_              = odt_cpu.ts_params_.m_2_;
        odt_gpu.ts_.forward_limit_    = odt_cpu.ts_params_.forward_limit_;
        odt_gpu.ts_.inverse_limit_    = odt_cpu.ts_params_.inverse_limit_;
        odt_gpu.ts_.log_peak_         = odt_cpu.ts_params_.log_peak_;
        odt_gpu.sat                   = odt_cpu.sat_;
        odt_gpu.sat_thr               = odt_cpu.sat_thr_;
        odt_gpu.compr                 = odt_cpu.compr_;
        odt_gpu.chroma_compress_scale = odt_cpu.chroma_compress_scale_;
        odt_gpu.mid_J                 = odt_cpu.mid_J_;
        odt_gpu.focus_dist            = odt_cpu.focus_dist_;
        odt_gpu.lower_hull_gamma_inv  = odt_cpu.lower_hull_gamma_inv_;
        odt_gpu.hue_linearity_search_range[0] =
            static_cast<int>(odt_cpu.hue_linearity_search_range_(0));
        odt_gpu.hue_linearity_search_range[1] =
            static_cast<int>(odt_cpu.hue_linearity_search_range_(1));
        copy_jmh(odt_cpu.input_params_, odt_gpu.input_params_);
        copy_jmh(odt_cpu.reach_params_, odt_gpu.reach_params_);
        copy_jmh(odt_cpu.limit_params_, odt_gpu.limit_params_);

        if (odt_cpu.table_reach_M_) {
          const std::uintptr_t id = reinterpret_cast<std::uintptr_t>(odt_cpu.table_reach_M_.get());
          if (id != odt_gpu.host_table_reach_M_id_) {
            odt_gpu.table_reach_M_.Reset();
            odt_gpu.table_reach_M_ = Create1DLinearTableTextureObject<float>(
                odt_cpu.table_reach_M_->data(), TOTAL_TABLE_SIZE);
            odt_gpu.host_table_reach_M_id_ = id;
          }
        }
        if (odt_cpu.table_hues_) {
          const std::uintptr_t id = reinterpret_cast<std::uintptr_t>(odt_cpu.table_hues_.get());
          if (id != odt_gpu.host_table_hues_id_) {
            odt_gpu.table_hues_.Reset();
            odt_gpu.table_hues_ = Create1DLinearTableTextureObject<float>(odt_cpu.table_hues_->data(),
                                                                          TOTAL_TABLE_SIZE);
            odt_gpu.host_table_hues_id_ = id;
          }
        }
        if (odt_cpu.table_upper_hull_gammas_) {
          const std::uintptr_t id =
              reinterpret_cast<std::uintptr_t>(odt_cpu.table_upper_hull_gammas_.get());
          if (id != odt_gpu.host_table_upper_hull_gamma_id_) {
            odt_gpu.table_upper_hull_gamma_.Reset();
            odt_gpu.table_upper_hull_gamma_ = Create1DLinearTableTextureObject<float>(
                odt_cpu.table_upper_hull_gammas_->data(), TOTAL_TABLE_SIZE);
            odt_gpu.host_table_upper_hull_gamma_id_ = id;
          }
        }
        if (odt_cpu.table_gamut_cusps_) {
          const std::uintptr_t id =
              reinterpret_cast<std::uintptr_t>(odt_cpu.table_gamut_cusps_.get());
          if (id != odt_gpu.host_table_gamut_cusps_id_) {
            odt_gpu.table_gamut_cusps_.Reset();
            std::vector<float4> packed(TOTAL_TABLE_SIZE);
            for (size_t i = 0; i < TOTAL_TABLE_SIZE; ++i) {
              const auto& v = (*odt_cpu.table_gamut_cusps_)[i];
              packed[i]     = make_float4(v(0), v(1), v(2), 0.f);
            }
            odt_gpu.table_gamut_cusps_ =
                Create1DLinearTableTextureObject<float4>(packed.data(), packed.size());
            odt_gpu.host_table_gamut_cusps_id_ = id;
          }
        }
        cpu_params.to_output_dirty_ = false;
      } else {
        to_output_gpu.aces_params_.Reset();
        const auto& open_cpu = to_output_cpu.open_drt_params_;
        auto&       open_gpu = to_output_gpu.open_drt_params_;
        open_gpu.tn_hcon_enable_ = open_cpu.tn_hcon_enable_;
        open_gpu.tn_lcon_enable_ = open_cpu.tn_lcon_enable_;
        open_gpu.pt_enable_      = open_cpu.pt_enable_;
        open_gpu.ptl_enable_     = open_cpu.ptl_enable_;
        open_gpu.ptm_enable_     = open_cpu.ptm_enable_;
        open_gpu.brl_enable_     = open_cpu.brl_enable_;
        open_gpu.brlp_enable_    = open_cpu.brlp_enable_;
        open_gpu.hc_enable_      = open_cpu.hc_enable_;
        open_gpu.hs_rgb_enable_  = open_cpu.hs_rgb_enable_;
        open_gpu.hs_cmy_enable_  = open_cpu.hs_cmy_enable_;
        open_gpu.creative_white_ = open_cpu.creative_white_;
        open_gpu.surround_       = open_cpu.surround_;
        open_gpu.clamp_          = open_cpu.clamp_;
        open_gpu.display_gamut_  = open_cpu.display_gamut_;
        open_gpu.display_eotf_   = open_cpu.display_eotf_;
        open_gpu.tn_con_         = open_cpu.tn_con_;
        open_gpu.tn_sh_          = open_cpu.tn_sh_;
        open_gpu.tn_toe_         = open_cpu.tn_toe_;
        open_gpu.tn_off_         = open_cpu.tn_off_;
        open_gpu.tn_hcon_        = open_cpu.tn_hcon_;
        open_gpu.tn_hcon_pv_     = open_cpu.tn_hcon_pv_;
        open_gpu.tn_hcon_st_     = open_cpu.tn_hcon_st_;
        open_gpu.tn_lcon_        = open_cpu.tn_lcon_;
        open_gpu.tn_lcon_w_      = open_cpu.tn_lcon_w_;
        open_gpu.cwp_lm_         = open_cpu.cwp_lm_;
        open_gpu.rs_sa_          = open_cpu.rs_sa_;
        open_gpu.rs_rw_          = open_cpu.rs_rw_;
        open_gpu.rs_bw_          = open_cpu.rs_bw_;
        open_gpu.pt_lml_         = open_cpu.pt_lml_;
        open_gpu.pt_lml_r_       = open_cpu.pt_lml_r_;
        open_gpu.pt_lml_g_       = open_cpu.pt_lml_g_;
        open_gpu.pt_lml_b_       = open_cpu.pt_lml_b_;
        open_gpu.pt_lmh_         = open_cpu.pt_lmh_;
        open_gpu.pt_lmh_r_       = open_cpu.pt_lmh_r_;
        open_gpu.pt_lmh_b_       = open_cpu.pt_lmh_b_;
        open_gpu.ptl_c_          = open_cpu.ptl_c_;
        open_gpu.ptl_m_          = open_cpu.ptl_m_;
        open_gpu.ptl_y_          = open_cpu.ptl_y_;
        open_gpu.ptm_low_        = open_cpu.ptm_low_;
        open_gpu.ptm_low_rng_    = open_cpu.ptm_low_rng_;
        open_gpu.ptm_low_st_     = open_cpu.ptm_low_st_;
        open_gpu.ptm_high_       = open_cpu.ptm_high_;
        open_gpu.ptm_high_rng_   = open_cpu.ptm_high_rng_;
        open_gpu.ptm_high_st_    = open_cpu.ptm_high_st_;
        open_gpu.brl_            = open_cpu.brl_;
        open_gpu.brl_r_          = open_cpu.brl_r_;
        open_gpu.brl_g_          = open_cpu.brl_g_;
        open_gpu.brl_b_          = open_cpu.brl_b_;
        open_gpu.brl_rng_        = open_cpu.brl_rng_;
        open_gpu.brl_st_         = open_cpu.brl_st_;
        open_gpu.brlp_           = open_cpu.brlp_;
        open_gpu.brlp_r_         = open_cpu.brlp_r_;
        open_gpu.brlp_g_         = open_cpu.brlp_g_;
        open_gpu.brlp_b_         = open_cpu.brlp_b_;
        open_gpu.hc_r_           = open_cpu.hc_r_;
        open_gpu.hc_r_rng_       = open_cpu.hc_r_rng_;
        open_gpu.hs_r_           = open_cpu.hs_r_;
        open_gpu.hs_r_rng_       = open_cpu.hs_r_rng_;
        open_gpu.hs_g_           = open_cpu.hs_g_;
        open_gpu.hs_g_rng_       = open_cpu.hs_g_rng_;
        open_gpu.hs_b_           = open_cpu.hs_b_;
        open_gpu.hs_b_rng_       = open_cpu.hs_b_rng_;
        open_gpu.hs_c_           = open_cpu.hs_c_;
        open_gpu.hs_c_rng_       = open_cpu.hs_c_rng_;
        open_gpu.hs_m_           = open_cpu.hs_m_;
        open_gpu.hs_m_rng_       = open_cpu.hs_m_rng_;
        open_gpu.hs_y_           = open_cpu.hs_y_;
        open_gpu.hs_y_rng_       = open_cpu.hs_y_rng_;
        open_gpu.ts_x1_          = open_cpu.ts_x1_;
        open_gpu.ts_y1_          = open_cpu.ts_y1_;
        open_gpu.ts_x0_          = open_cpu.ts_x0_;
        open_gpu.ts_y0_          = open_cpu.ts_y0_;
        open_gpu.ts_s0_          = open_cpu.ts_s0_;
        open_gpu.ts_p_           = open_cpu.ts_p_;
        open_gpu.ts_s10_         = open_cpu.ts_s10_;
        open_gpu.ts_m1_          = open_cpu.ts_m1_;
        open_gpu.ts_m2_          = open_cpu.ts_m2_;
        open_gpu.ts_s_           = open_cpu.ts_s_;
        open_gpu.ts_dsc_         = open_cpu.ts_dsc_;
        open_gpu.pt_cmp_Lf_      = open_cpu.pt_cmp_Lf_;
        open_gpu.s_Lp100_        = open_cpu.s_Lp100_;
        open_gpu.ts_s1_          = open_cpu.ts_s1_;
        cpu_params.to_output_dirty_ = false;
      }
    } else {
      gpu_params.to_output_params_ = orig_resources.uploaded_params_.to_output_params_;
    }

    return resources;
  }

 private:
  struct LUTPathCache {
    std::mutex                              mutex_;
    std::unordered_map<std::string, GPU_LUT3D> entries_;
  };

  static auto GetLUTPathCache() -> LUTPathCache& {
    // Intentionally leaked to avoid CUDA shutdown ordering issues during static destruction.
    static LUTPathCache* cache = new LUTPathCache();
    return *cache;
  }

  static auto BuildLUTPathCacheKey(const std::filesystem::path& path) -> std::string {
    std::error_code ec;
    const auto      abs_path   = std::filesystem::absolute(path, ec);
    const auto      normalized = (ec ? path : abs_path).lexically_normal();

    std::string     key        = normalized.string();

    std::error_code size_ec;
    const auto      file_size  = std::filesystem::file_size(normalized, size_ec);
    key += "|s=" + std::to_string(size_ec ? static_cast<std::uintmax_t>(0) : file_size);

    std::error_code time_ec;
    const auto      mtime      = std::filesystem::last_write_time(normalized, time_ec);
    const auto      stamp      = time_ec ? 0LL : mtime.time_since_epoch().count();
    key += "|t=" + std::to_string(stamp);

    return key;
  }

  static auto BorrowLUT3D(const GPU_LUT3D& lut) -> GPU_LUT3D {
    GPU_LUT3D borrowed = lut;
    borrowed.borrowed_ = true;
    return borrowed;
  }

  static GPU_LUT3D CreateLUTTextureObject(const std::vector<float>& lut_data, int edge_size) {
    GPU_LUT3D gpu_lut;
    gpu_lut.edge_size_         = edge_size;

    const size_t        voxels = static_cast<size_t>(edge_size) * edge_size * edge_size;
    std::vector<float4> packed(voxels);
    for (size_t i = 0; i < voxels; ++i) {
      packed[i] = make_float4(lut_data[i * 3 + 0], lut_data[i * 3 + 1], lut_data[i * 3 + 2], 1.0f);
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaExtent            extent       = make_cudaExtent(edge_size, edge_size, edge_size);
    cudaMalloc3DArray(&gpu_lut.array_, &channel_desc, extent);

    cudaMemcpy3DParms copy_params = {0};
    copy_params.srcPtr =
        make_cudaPitchedPtr((void*)packed.data(), edge_size * sizeof(float4), edge_size, edge_size);
    copy_params.dstArray = gpu_lut.array_;
    copy_params.extent   = extent;
    copy_params.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy_params);

    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = gpu_lut.array_;

    cudaTextureDesc tex_desc  = {};
    tex_desc.normalizedCoords = 1;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;

    cudaCreateTextureObject(&gpu_lut.texture_object_, &res_desc, &tex_desc, nullptr);
    return gpu_lut;
  };

  static GPU_LUT3D CreateLUTTextureObject(OCIO::ConstBakerRcPtr baker) {
    std::ostringstream oss;
    baker->bake(oss);
    // Parse LUT data from lut_str and create a 3D texture object
    CubeLut lut;
    ParseCubeString(oss.str(), lut);

    // Create CUDA 3D texture object from LUT data
    if (lut.Has3D()) {
      return CreateLUTTextureObject(lut.lut3d_, lut.edge3d_);
    }

    // TODO: Add support for 1D LUTs if needed
    throw std::runtime_error("GPUParamsConverter: Only 3D LUTs are supported for GPU processing.");
  };

  static GPU_LUT3D CreateLUTTextureObject(const std::filesystem::path& path) {
    const std::string cache_key = BuildLUTPathCacheKey(path);
    {
      auto&                      cache = GetLUTPathCache();
      std::lock_guard<std::mutex> lock(cache.mutex_);
      const auto                 it = cache.entries_.find(cache_key);
      if (it != cache.entries_.end()) {
        return BorrowLUT3D(it->second);
      }
    }

    CubeLut     lut;
    std::string parse_error;
    if (!ParseCubeFile(path, lut, &parse_error)) {
      std::ostringstream oss;
      oss << "GPUParamsConverter: Failed to parse LUT file '" << path.string()
          << "': " << parse_error;
      throw std::runtime_error(oss.str());
    }
    if (!lut.Has3D()) {
      // TODO: Add support for 1D LUTs if needed
      throw std::runtime_error(
          "GPUParamsConverter: Only 3D LUTs are supported for GPU processing.");
    }

    GPU_LUT3D parsed_lut = CreateLUTTextureObject(lut.lut3d_, lut.edge3d_);

    auto&                      cache = GetLUTPathCache();
    std::lock_guard<std::mutex> lock(cache.mutex_);
    auto                       [it, inserted] = cache.entries_.emplace(cache_key, parsed_lut);
    if (!inserted) {
      // Another thread populated the same key first; release our duplicate upload.
      parsed_lut.Reset();
    }
    return BorrowLUT3D(it->second);
  };
};

class GPUParamsConverter {
 public:
  static auto ConvertFromCPU(OperatorParams& cpu_params,
                             GPUOperatorParams& orig_params) -> GPUOperatorParams {
    FusedOperatorParams fused = FusedParamsConverter::ConvertFromCPU(cpu_params);
    CudaFusedResources  resources;
    resources.uploaded_params_ = orig_params;
    resources                  = CudaFusedParamUploader::Upload(fused, cpu_params, resources);
    return resources.uploaded_params_;
  }
};
};  // namespace puerhlab
