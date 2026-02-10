//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "edit/operators/op_base.hpp"
#include "utils/lut/cube_lut.hpp"

#define GPU_FUNC __device__ __forceinline__

namespace puerhlab {
enum class GPU_ETOF : int {
  LINEAR    = 0,
  ST2084    = 1,
  HLG       = 2,
  GAMMA_2_6 = 3,
  BT1886    = 4,
  GAMMA_2_2 = 5,
  GAMMA_1_8 = 6,
};

struct GPU_LUT3D {
  cudaArray_t         array_              = nullptr;
  cudaTextureObject_t texture_object_     = 0;
  int                 edge_size_          = 0;

  GPU_LUT3D()                             = default;

  GPU_LUT3D(const GPU_LUT3D&)             = default;

  GPU_LUT3D& operator=(const GPU_LUT3D&)  = default;

  GPU_LUT3D(GPU_LUT3D&& other)            = default;

  GPU_LUT3D& operator=(GPU_LUT3D&& other) = default;

  void       Reset() {
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

struct GPU_TO_OUTPUT_Params {
  GPU_ODTParams odt_params_ = {};

  float         limit_to_display_matx[9];
  GPU_ETOF      etof = GPU_ETOF::LINEAR;
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

  // White and Black point adjustment parameters
  bool                 white_enabled_          = true;
  float                white_point_            = 1.0f;

  bool                 black_enabled_          = true;
  float                black_point_            = 0.0f;

  float                slope_                  = 1.0f;
  // HLS adjustment parameters
  bool                 hls_enabled_            = true;
  float                target_hls_[3]          = {0.0f, 0.0f, 0.0f};
  float                hls_adjustment_[3]      = {0.0f, 0.0f, 0.0f};
  float                hue_range_              = 0.0f;
  float                lightness_range_        = 0.0f;
  float                saturation_range_       = 0.0f;

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

  // Sharpen adjustment parameter
  bool                 sharpen_enabled_        = true;
  float                sharpen_offset_         = 0.0f;
  float                sharpen_radius_         = 3.0f;
  float                sharpen_threshold_      = 0.0f;

  // Color wheel adjustment parameters
  bool                 color_wheel_enabled_    = true;
  float                lift_color_offset_[3]   = {0.0f, 0.0f, 0.0f};
  float                lift_luminance_offset_  = 0.0f;
  float                gamma_color_offset_[3]  = {1.0f, 1.0f, 1.0f};
  float                gamma_luminance_offset_ = 0.0f;
  float                gain_color_offset_[3]   = {1.0f, 1.0f, 1.0f};
  float                gain_luminance_offset_  = 0.0f;
};

class GPUParamsConverter {
 public:
  static GPUOperatorParams ConvertFromCPU(OperatorParams&    cpu_params,
                                          GPUOperatorParams& orig_params) {
    // TODO: Improve param synchronization to avoid unnecessary data transfers
    GPUOperatorParams gpu_params       = orig_params;

    gpu_params.exposure_enabled_       = cpu_params.exposure_enabled_;
    gpu_params.exposure_offset_        = cpu_params.exposure_offset_;

    gpu_params.contrast_enabled_       = cpu_params.contrast_enabled_;
    gpu_params.contrast_scale_         = cpu_params.contrast_scale_;

    gpu_params.shadows_enabled_        = cpu_params.shadows_enabled_;
    gpu_params.shadows_offset_         = cpu_params.shadows_offset_;
    gpu_params.shadows_x0_             = cpu_params.shadows_x0_;
    gpu_params.shadows_x1_             = cpu_params.shadows_x1_;
    gpu_params.shadows_y0_             = cpu_params.shadows_y0_;
    gpu_params.shadows_y1_             = cpu_params.shadows_y1_;
    gpu_params.shadows_m0_             = cpu_params.shadows_m0_;
    gpu_params.shadows_m1_             = cpu_params.shadows_m1_;
    gpu_params.shadows_dx_             = cpu_params.shadows_dx_;

    gpu_params.highlights_enabled_     = cpu_params.highlights_enabled_;
    gpu_params.highlights_k_           = cpu_params.highlights_k_;
    gpu_params.highlights_offset_      = cpu_params.highlights_offset_;
    gpu_params.highlights_slope_range_ = cpu_params.highlights_slope_range_;
    gpu_params.highlights_m0_          = cpu_params.highlights_m0_;
    gpu_params.highlights_m1_          = cpu_params.highlights_m1_;
    gpu_params.highlights_x0_          = cpu_params.highlights_x0_;
    gpu_params.highlights_y0_          = cpu_params.highlights_y0_;
    gpu_params.highlights_y1_          = cpu_params.highlights_y1_;
    gpu_params.highlights_dx_          = cpu_params.highlights_dx_;

    gpu_params.white_enabled_          = cpu_params.white_enabled_;
    gpu_params.white_point_            = cpu_params.white_point_;

    gpu_params.black_enabled_          = cpu_params.black_enabled_;
    gpu_params.black_point_            = cpu_params.black_point_;

    gpu_params.slope_                  = cpu_params.slope_;

    gpu_params.hls_enabled_            = cpu_params.hls_enabled_;
    for (int i = 0; i < 3; ++i) {
      gpu_params.target_hls_[i]     = cpu_params.target_hls_[i];
      gpu_params.hls_adjustment_[i] = cpu_params.hls_adjustment_[i];
    }
    gpu_params.hue_range_          = cpu_params.hue_range_;
    gpu_params.lightness_range_    = cpu_params.lightness_range_;

    gpu_params.saturation_range_   = cpu_params.saturation_range_;
    gpu_params.saturation_enabled_ = cpu_params.saturation_enabled_;
    gpu_params.saturation_offset_  = cpu_params.saturation_offset_;

    gpu_params.tint_enabled_       = cpu_params.tint_enabled_;
    gpu_params.tint_offset_        = cpu_params.tint_offset_;

    gpu_params.vibrance_enabled_   = cpu_params.vibrance_enabled_;
    gpu_params.vibrance_offset_    = cpu_params.vibrance_offset_;

    gpu_params.to_ws_enabled_      = cpu_params.to_ws_enabled_;
    // if (cpu_params.to_ws_dirty) {
    //   gpu_params.to_ws_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_ws_lut        = CreateLUTTextureObject(cpu_params.to_ws_lut_baker);
    //   cpu_params.to_ws_dirty      = false;
    // } else {
    //   gpu_params.to_ws_lut = orig_params.to_ws_lut;
    // }

    gpu_params.lmt_enabled_        = cpu_params.lmt_enabled_;
    const bool lmt_gpu_lut_missing =
        (orig_params.lmt_lut_.texture_object_ == 0 || orig_params.lmt_lut_.edge_size_ <= 1);
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
      gpu_params.lmt_lut_ = orig_params.lmt_lut_;
    }

    gpu_params.to_output_enabled_   = cpu_params.to_output_enabled_;
    // if (cpu_params.to_output_dirty) {
    //   gpu_params.to_output_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_output_lut        = CreateLUTTextureObject(cpu_params.to_output_lut_baker);
    //   cpu_params.to_output_dirty      = false;
    // } else {
    //   gpu_params.to_output_lut = orig_params.to_output_lut;
    // }

    gpu_params.curve_enabled_       = cpu_params.curve_enabled_;
    const size_t curve_pts_count    = cpu_params.curve_ctrl_pts_.size();
    if (curve_pts_count > static_cast<size_t>(GPUOperatorParams::kMaxCurveControlPoints)) {
      std::ostringstream oss;
      oss << "GPUParamsConverter: curve has " << curve_pts_count
          << " control points, but GPU max is " << GPUOperatorParams::kMaxCurveControlPoints << ".";
      throw std::runtime_error(oss.str());
    }
    if (curve_pts_count > 0 && cpu_params.curve_m_.size() < curve_pts_count) {
      throw std::runtime_error(
          "GPUParamsConverter: curve_m_ is smaller than curve control-point count.");
    }
    if (curve_pts_count > 1 && cpu_params.curve_h_.size() < (curve_pts_count - 1)) {
      throw std::runtime_error(
          "GPUParamsConverter: curve_h_ is smaller than curve segment count.");
    }

    gpu_params.curve_ctrl_pts_size_ = static_cast<int>(curve_pts_count);
    for (int i = 0; i < GPUOperatorParams::kMaxCurveControlPoints; ++i) {
      gpu_params.curve_ctrl_pts_x_[i] = 0.0f;
      gpu_params.curve_ctrl_pts_y_[i] = 0.0f;
      gpu_params.curve_m_[i]          = 0.0f;
      if (i < GPUOperatorParams::kMaxCurveControlPoints - 1) {
        gpu_params.curve_h_[i] = 0.0f;
      }
    }
    for (size_t i = 0; i < curve_pts_count; ++i) {
      gpu_params.curve_ctrl_pts_x_[i] = cpu_params.curve_ctrl_pts_[i].x;
      gpu_params.curve_ctrl_pts_y_[i] = cpu_params.curve_ctrl_pts_[i].y;
      gpu_params.curve_m_[i]          = cpu_params.curve_m_[i];
    }
    for (size_t i = 0; i + 1 < curve_pts_count; ++i) {
      gpu_params.curve_h_[i] = cpu_params.curve_h_[i];
    }

    gpu_params.clarity_enabled_     = cpu_params.clarity_enabled_;
    gpu_params.clarity_offset_      = cpu_params.clarity_offset_;
    gpu_params.clarity_radius_      = cpu_params.clarity_radius_;

    gpu_params.sharpen_enabled_     = cpu_params.sharpen_enabled_;
    gpu_params.sharpen_offset_      = cpu_params.sharpen_offset_;
    gpu_params.sharpen_radius_      = cpu_params.sharpen_radius_;
    gpu_params.sharpen_threshold_   = cpu_params.sharpen_threshold_;

    gpu_params.color_wheel_enabled_ = cpu_params.color_wheel_enabled_;
    for (int i = 0; i < 3; ++i) {
      gpu_params.lift_color_offset_[i]  = cpu_params.lift_color_offset_[i];
      gpu_params.gamma_color_offset_[i] = cpu_params.gamma_color_offset_[i];
      gpu_params.gain_color_offset_[i]  = cpu_params.gain_color_offset_[i];
    }
    gpu_params.lift_luminance_offset_  = cpu_params.lift_luminance_offset_;
    gpu_params.gamma_luminance_offset_ = cpu_params.gamma_luminance_offset_;
    gpu_params.gain_luminance_offset_  = cpu_params.gain_luminance_offset_;

    // ----------------------------------------------------------------------
    // ODT params (CTL port): copy CPU-side precomputed tables into
    // per-instance CUDA resources. This keeps the GPU path stateless and safe
    // for future multi-frame / multi-stream parallelism.
    //
    // We cache uploads by tracking the address of the shared_ptr backing arrays.
    // Current CPU implementation regenerates these tables by allocating new
    // shared_ptrs, so pointer changes imply new contents.
    // ----------------------------------------------------------------------
    if (cpu_params.to_output_dirty_) {
      auto&       to_output_gpu = gpu_params.to_output_params_;
      const auto& to_output_cpu = cpu_params.to_output_params_;

      // Fail fast if ODT precompute is missing.
      // Without these tables, the GPU ODT path will end up calling tex1Dfetch
      // with texture_object_=0, which is undefined and can manifest as
      // run-to-run nondeterminism.
      if (cpu_params.to_output_enabled_) {
        const auto& odt_check = cpu_params.to_output_params_.odt_params_;
        const bool  missing_tables = (!odt_check.table_reach_M_) || (!odt_check.table_hues_) ||
                                    (!odt_check.table_upper_hull_gammas_) ||
                                    (!odt_check.table_gamut_cusps_);
        if (missing_tables) {
          std::ostringstream oss;
          oss << "GPUParamsConverter: ODT tables are not initialized:";
          if (!odt_check.table_reach_M_) oss << " table_reach_M_";
          if (!odt_check.table_hues_) oss << " table_hues_";
          if (!odt_check.table_upper_hull_gammas_) oss << " table_upper_hull_gammas_";
          if (!odt_check.table_gamut_cusps_) oss << " table_gamut_cusps_";
          oss << ". Ensure CPU ODT precompute runs before executing the GPU pipeline.";
          throw std::runtime_error(oss.str());
        }
      }
      // Matrices: flatten cv::Matx33f into row-major float[9]
      auto        copy33        = [](const cv::Matx33f& m, float out[9]) {
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

      // Copy TO_OUTPUT params
      // Limit to display matrix
      copy33(to_output_cpu.limit_to_display_matx_, to_output_gpu.limit_to_display_matx);
      // ETOF
      // These two enums have identical values; static_cast through int to be safe
      to_output_gpu.etof = static_cast<GPU_ETOF>(static_cast<int>(to_output_cpu.etof_));

      const auto& odt_cpu           = cpu_params.to_output_params_.odt_params_;
      auto&       odt_gpu           = gpu_params.to_output_params_.odt_params_;

      odt_gpu.peak_luminance_       = odt_cpu.peak_luminance_;
      odt_gpu.limit_J_max           = odt_cpu.limit_J_max_;
      odt_gpu.model_gamma_inv       = odt_cpu.model_gamma_inv_;

      // Tone scale
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

      // Chroma compression
      odt_gpu.sat                   = odt_cpu.sat_;
      odt_gpu.sat_thr               = odt_cpu.sat_thr_;
      odt_gpu.compr                 = odt_cpu.compr_;
      odt_gpu.chroma_compress_scale = odt_cpu.chroma_compress_scale_;

      // Gamut compression
      odt_gpu.mid_J                 = odt_cpu.mid_J_;
      odt_gpu.focus_dist            = odt_cpu.focus_dist_;
      odt_gpu.lower_hull_gamma_inv  = odt_cpu.lower_hull_gamma_inv_;
      odt_gpu.hue_linearity_search_range[0] =
          static_cast<int>(odt_cpu.hue_linearity_search_range_(0));
      odt_gpu.hue_linearity_search_range[1] =
          static_cast<int>(odt_cpu.hue_linearity_search_range_(1));

      auto copy_jmh = [&](const ColorUtils::JMhParams& src, GPU_JMhParams& dst) {
        copy33(src.MATRIX_RGB_to_CAM16_c_, dst.MATRIX_RGB_to_CAM16_c_);
        copy33(src.MATRIX_CAM16_c_to_RGB_, dst.MATRIX_CAM16_c_to_RGB_);
        copy33(src.MATRIX_cone_response_to_Aab_, dst.MATRIX_cone_response_to_Aab_);
        copy33(src.MATRIX_Aab_to_cone_response_, dst.MATRIX_Aab_to_cone_response_);
        dst.F_L_n_     = src.F_L_n_;
        dst.cz_        = src.cz_;
        dst.inv_cz_    = src.inv_cz_;
        // CTL semantics: A_w_J = _pacrc_fwd(F_L). Use inv_A_w_J_ as the
        // authoritative value to avoid any CPU-side naming/meaning mismatches.
        dst.A_w_J_     = (src.inv_A_w_J_ != 0.f) ? (1.f / src.inv_A_w_J_) : 0.f;
        dst.inv_A_w_J_ = src.inv_A_w_J_;
      };

      copy_jmh(odt_cpu.input_params_, odt_gpu.input_params_);
      copy_jmh(odt_cpu.reach_params_, odt_gpu.reach_params_);
      copy_jmh(odt_cpu.limit_params_, odt_gpu.limit_params_);

      // Upload tables (cached by shared_ptr backing address)
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
      gpu_params.to_output_params_ = orig_params.to_output_params_;
    }

    return gpu_params;
  }

 private:
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

  static GPU_LUT3D CreateLUTTextureObject(std::filesystem::path& path) {
    CubeLut lut;
    ParseCubeFile(path, lut);
    if (lut.Has3D()) {
      return CreateLUTTextureObject(lut.lut3d_, lut.edge3d_);
    }
    // TODO: Add support for 1D LUTs if needed
    throw std::runtime_error("GPUParamsConverter: Only 3D LUTs are supported for GPU processing.");
  };
};
};  // namespace puerhlab
