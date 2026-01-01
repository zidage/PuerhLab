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

#include <sstream>

#include "edit/operators/op_base.hpp"
#include "utils/lut/cube_lut.hpp"

namespace puerhlab {
struct GPU_LUT3D {
  cudaArray_t         array_               = nullptr;
  cudaTextureObject_t texture_object_      = 0;
  int                 edge_size_           = 0;

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

struct GPUOperatorParams {
  // Basic adjustment parameters
  bool      exposure_enabled_       = true;
  float     exposure_offset_        = 0.0f;

  bool      contrast_enabled_       = true;
  float     contrast_scale_         = 0.0f;

  // Shadows adjustment parameter
  bool      shadows_enabled_        = true;
  float     shadows_offset_         = 0.0f;
  float     shadows_x0_             = 0.0f;
  float     shadows_x1_             = 0.25f;
  float     shadows_y0_             = 0.0f;
  float     shadows_y1_             = 0.25f;
  float     shadows_m0_             = 0.0f;
  float     shadows_m1_             = 1.0f;
  float     shadows_dx_             = 0.25f;

  // Highlights adjustment parameter
  bool      highlights_enabled_     = true;
  float     highlights_k_           = 0.2f;
  float     highlights_offset_      = 0.0f;
  float     highlights_slope_range_ = 0.8f;
  float     highlights_m0_          = 1.0f;
  float     highlights_m1_          = 1.0f;
  float     highlights_x0_          = 0.2f;
  float     highlights_y0_          = 0.2f;
  float     highlights_y1_          = 1.0f;
  float     highlights_dx_          = 0.8f;

  // White and Black point adjustment parameters
  bool      white_enabled_          = true;
  float     white_point_            = 1.0f;

  bool      black_enabled_          = true;
  float     black_point_            = 0.0f;

  float     slope_                  = 1.0f;
  // HLS adjustment parameters
  bool      hls_enabled_            = true;
  float     target_hls_[3]          = {0.0f, 0.0f, 0.0f};
  float     hls_adjustment_[3]      = {0.0f, 0.0f, 0.0f};
  float     hue_range_              = 0.0f;
  float     lightness_range_        = 0.0f;
  float     saturation_range_       = 0.0f;

  // Saturation adjustment parameter
  bool      saturation_enabled_     = true;
  float     saturation_offset_      = 0.0f;

  // Tint adjustment parameter
  bool      tint_enabled_           = true;
  float     tint_offset_            = 0.0f;

  // Vibrance adjustment parameter
  bool      vibrance_enabled_       = true;
  float     vibrance_offset_        = 0.0f;

  // Working space
  bool      to_ws_enabled_          = true;
  GPU_LUT3D to_ws_lut_              = {};
  // TODO: NOT IMPLEMENTED

  // Look modification transform
  bool      lmt_enabled_            = false;
  GPU_LUT3D lmt_lut_                = {};
  // TODO: NOT IMPLEMENTED

  // Output transform
  bool      to_output_enabled_      = true;
  GPU_LUT3D to_output_lut_          = {};

  // Curve adjustment parameters
  bool      curve_enabled_          = false;

  // Clarity adjustment parameter
  bool      clarity_enabled_        = true;
  float     clarity_offset_         = 0.0f;
  float     clarity_radius_         = 5.0f;

  // Sharpen adjustment parameter
  bool      sharpen_enabled_        = true;
  float     sharpen_offset_         = 0.0f;
  float     sharpen_radius_         = 3.0f;
  float     sharpen_threshold_      = 0.0f;

  // Color wheel adjustment parameters
  bool      color_wheel_enabled_    = true;
  float     lift_color_offset_[3]   = {0.0f, 0.0f, 0.0f};
  float     lift_luminance_offset_  = 0.0f;
  float     gamma_color_offset_[3]  = {1.0f, 1.0f, 1.0f};
  float     gamma_luminance_offset_ = 0.0f;
  float     gain_color_offset_[3]   = {1.0f, 1.0f, 1.0f};
  float     gain_luminance_offset_  = 0.0f;
};

class GPUParamsConverter {
 public:
  static GPUOperatorParams ConvertFromCPU(OperatorParams&    cpu_params,
                                          GPUOperatorParams& orig_params) {
    // TODO: Improve param synchronization to avoid unnecessary data transfers
    GPUOperatorParams gpu_params;

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

    gpu_params.lmt_enabled_ = cpu_params.lmt_enabled_;
    if (cpu_params.to_lmt_dirty_) {
      gpu_params.lmt_lut_.Reset();  // Explicitly reset existing LUT
      gpu_params.lmt_lut_           = CreateLUTTextureObject(cpu_params.lmt_lut_path_);
      cpu_params.to_lmt_dirty_      = false;
    } else {
      gpu_params.lmt_lut_ = orig_params.lmt_lut_;
    }

    gpu_params.to_output_enabled_ = cpu_params.to_output_enabled_;
    // if (cpu_params.to_output_dirty) {
    //   gpu_params.to_output_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_output_lut        = CreateLUTTextureObject(cpu_params.to_output_lut_baker);
    //   cpu_params.to_output_dirty      = false;
    // } else {
    //   gpu_params.to_output_lut = orig_params.to_output_lut;
    // }

    gpu_params.curve_enabled_       = cpu_params.curve_enabled_;

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

    return gpu_params;
  }

 private:
  static GPU_LUT3D CreateLUTTextureObject(const std::vector<float>& lut_data, int edge_size) {
    GPU_LUT3D gpu_lut;
    gpu_lut.edge_size_          = edge_size;

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