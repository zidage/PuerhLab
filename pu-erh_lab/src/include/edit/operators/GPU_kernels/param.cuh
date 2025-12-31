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
  cudaArray_t         array               = nullptr;
  cudaTextureObject_t texture_object      = 0;
  int                 edge_size           = 0;

  GPU_LUT3D()                             = default;

  GPU_LUT3D(const GPU_LUT3D&)             = default;

  GPU_LUT3D& operator=(const GPU_LUT3D&)  = default;

  GPU_LUT3D(GPU_LUT3D&& other)            = default;

  GPU_LUT3D& operator=(GPU_LUT3D&& other) = default;

  void       Reset() {
    if (texture_object) {
      cudaDestroyTextureObject(texture_object);
      texture_object = 0;
    }
    if (array) {
      cudaFreeArray(array);
      array = nullptr;
    }
    edge_size = 0;
  }
};

struct GPUOperatorParams {
  // Basic adjustment parameters
  bool      exposure_enabled       = true;
  float     exposure_offset        = 0.0f;

  bool      contrast_enabled       = true;
  float     contrast_scale         = 0.0f;

  // Shadows adjustment parameter
  bool      shadows_enabled        = true;
  float     shadows_offset         = 0.0f;
  float     shadows_x0             = 0.0f;
  float     shadows_x1             = 0.25f;
  float     shadows_y0             = 0.0f;
  float     shadows_y1             = 0.25f;
  float     shadows_m0             = 0.0f;
  float     shadows_m1             = 1.0f;
  float     shadows_dx             = 0.25f;

  // Highlights adjustment parameter
  bool      highlights_enabled     = true;
  float     highlights_k           = 0.2f;
  float     highlights_offset      = 0.0f;
  float     highlights_slope_range = 0.8f;
  float     highlights_m0          = 1.0f;
  float     highlights_m1          = 1.0f;
  float     highlights_x0          = 0.2f;
  float     highlights_y0          = 0.2f;
  float     highlights_y1          = 1.0f;
  float     highlights_dx          = 0.8f;

  // White and Black point adjustment parameters
  bool      white_enabled          = true;
  float     white_point            = 1.0f;

  bool      black_enabled          = true;
  float     black_point            = 0.0f;

  float     slope                  = 1.0f;
  // HLS adjustment parameters
  bool      hls_enabled            = true;
  float     target_hls[3]          = {0.0f, 0.0f, 0.0f};
  float     hls_adjustment[3]      = {0.0f, 0.0f, 0.0f};
  float     hue_range              = 0.0f;
  float     lightness_range        = 0.0f;
  float     saturation_range       = 0.0f;

  // Saturation adjustment parameter
  bool      saturation_enabled     = true;
  float     saturation_offset      = 0.0f;

  // Tint adjustment parameter
  bool      tint_enabled           = true;
  float     tint_offset            = 0.0f;

  // Vibrance adjustment parameter
  bool      vibrance_enabled       = true;
  float     vibrance_offset        = 0.0f;

  // Working space
  bool      to_ws_enabled          = true;
  GPU_LUT3D to_ws_lut              = {};
  // TODO: NOT IMPLEMENTED

  // Look modification transform
  bool      lmt_enabled            = false;
  GPU_LUT3D lmt_lut                = {};
  // TODO: NOT IMPLEMENTED

  // Output transform
  bool      to_output_enabled      = true;
  GPU_LUT3D to_output_lut          = {};

  // Curve adjustment parameters
  bool      curve_enabled          = false;

  // Clarity adjustment parameter
  bool      clarity_enabled        = true;
  float     clarity_offset         = 0.0f;
  float     clarity_radius         = 5.0f;

  // Sharpen adjustment parameter
  bool      sharpen_enabled        = true;
  float     sharpen_offset         = 0.0f;
  float     sharpen_radius         = 3.0f;
  float     sharpen_threshold      = 0.0f;

  // Color wheel adjustment parameters
  bool      color_wheel_enabled    = true;
  float     lift_color_offset[3]   = {0.0f, 0.0f, 0.0f};
  float     lift_luminance_offset  = 0.0f;
  float     gamma_color_offset[3]  = {1.0f, 1.0f, 1.0f};
  float     gamma_luminance_offset = 0.0f;
  float     gain_color_offset[3]   = {1.0f, 1.0f, 1.0f};
  float     gain_luminance_offset  = 0.0f;
};

class GPUParamsConverter {
 public:
  static GPUOperatorParams ConvertFromCPU(OperatorParams&    cpu_params,
                                          GPUOperatorParams& orig_params) {
    // TODO: Improve param synchronization to avoid unnecessary data transfers
    GPUOperatorParams gpu_params;

    gpu_params.exposure_enabled       = cpu_params.exposure_enabled;
    gpu_params.exposure_offset        = cpu_params.exposure_offset;

    gpu_params.contrast_enabled       = cpu_params.contrast_enabled;
    gpu_params.contrast_scale         = cpu_params.contrast_scale;

    gpu_params.shadows_enabled        = cpu_params.shadows_enabled;
    gpu_params.shadows_offset         = cpu_params.shadows_offset;
    gpu_params.shadows_x0             = cpu_params.shadows_x0;
    gpu_params.shadows_x1             = cpu_params.shadows_x1;
    gpu_params.shadows_y0             = cpu_params.shadows_y0;
    gpu_params.shadows_y1             = cpu_params.shadows_y1;
    gpu_params.shadows_m0             = cpu_params.shadows_m0;
    gpu_params.shadows_m1             = cpu_params.shadows_m1;
    gpu_params.shadows_dx             = cpu_params.shadows_dx;

    gpu_params.highlights_enabled     = cpu_params.highlights_enabled;
    gpu_params.highlights_k           = cpu_params.highlights_k;
    gpu_params.highlights_offset      = cpu_params.highlights_offset;
    gpu_params.highlights_slope_range = cpu_params.highlights_slope_range;
    gpu_params.highlights_m0          = cpu_params.highlights_m0;
    gpu_params.highlights_m1          = cpu_params.highlights_m1;
    gpu_params.highlights_x0          = cpu_params.highlights_x0;
    gpu_params.highlights_y0          = cpu_params.highlights_y0;
    gpu_params.highlights_y1          = cpu_params.highlights_y1;
    gpu_params.highlights_dx          = cpu_params.highlights_dx;

    gpu_params.white_enabled          = cpu_params.white_enabled;
    gpu_params.white_point            = cpu_params.white_point;

    gpu_params.black_enabled          = cpu_params.black_enabled;
    gpu_params.black_point            = cpu_params.black_point;

    gpu_params.slope                  = cpu_params.slope;

    gpu_params.hls_enabled            = cpu_params.hls_enabled;
    for (int i = 0; i < 3; ++i) {
      gpu_params.target_hls[i]     = cpu_params.target_hls[i];
      gpu_params.hls_adjustment[i] = cpu_params.hls_adjustment[i];
    }
    gpu_params.hue_range          = cpu_params.hue_range;
    gpu_params.lightness_range    = cpu_params.lightness_range;

    gpu_params.saturation_range   = cpu_params.saturation_range;
    gpu_params.saturation_enabled = cpu_params.saturation_enabled;
    gpu_params.saturation_offset  = cpu_params.saturation_offset;

    gpu_params.tint_enabled       = cpu_params.tint_enabled;
    gpu_params.tint_offset        = cpu_params.tint_offset;

    gpu_params.vibrance_enabled   = cpu_params.vibrance_enabled;
    gpu_params.vibrance_offset    = cpu_params.vibrance_offset;

    gpu_params.to_ws_enabled      = cpu_params.to_ws_enabled;
    // if (cpu_params.to_ws_dirty) {
    //   gpu_params.to_ws_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_ws_lut        = CreateLUTTextureObject(cpu_params.to_ws_lut_baker);
    //   cpu_params.to_ws_dirty      = false;
    // } else {
    //   gpu_params.to_ws_lut = orig_params.to_ws_lut;
    // }

    gpu_params.lmt_enabled = cpu_params.lmt_enabled;
    if (cpu_params.to_lmt_dirty) {
      gpu_params.lmt_lut.Reset();  // Explicitly reset existing LUT
      gpu_params.lmt_lut           = CreateLUTTextureObject(cpu_params.lmt_lut_path);
      cpu_params.to_lmt_dirty      = false;
    } else {
      gpu_params.lmt_lut = orig_params.lmt_lut;
    }

    gpu_params.to_output_enabled = cpu_params.to_output_enabled;
    // if (cpu_params.to_output_dirty) {
    //   gpu_params.to_output_lut.Reset();  // Explicitly reset existing LUT
    //   gpu_params.to_output_lut        = CreateLUTTextureObject(cpu_params.to_output_lut_baker);
    //   cpu_params.to_output_dirty      = false;
    // } else {
    //   gpu_params.to_output_lut = orig_params.to_output_lut;
    // }

    gpu_params.curve_enabled       = cpu_params.curve_enabled;

    gpu_params.clarity_enabled     = cpu_params.clarity_enabled;
    gpu_params.clarity_offset      = cpu_params.clarity_offset;
    gpu_params.clarity_radius      = cpu_params.clarity_radius;

    gpu_params.sharpen_enabled     = cpu_params.sharpen_enabled;
    gpu_params.sharpen_offset      = cpu_params.sharpen_offset;
    gpu_params.sharpen_radius      = cpu_params.sharpen_radius;
    gpu_params.sharpen_threshold   = cpu_params.sharpen_threshold;

    gpu_params.color_wheel_enabled = cpu_params.color_wheel_enabled;
    for (int i = 0; i < 3; ++i) {
      gpu_params.lift_color_offset[i]  = cpu_params.lift_color_offset[i];
      gpu_params.gamma_color_offset[i] = cpu_params.gamma_color_offset[i];
      gpu_params.gain_color_offset[i]  = cpu_params.gain_color_offset[i];
    }
    gpu_params.lift_luminance_offset  = cpu_params.lift_luminance_offset;
    gpu_params.gamma_luminance_offset = cpu_params.gamma_luminance_offset;
    gpu_params.gain_luminance_offset  = cpu_params.gain_luminance_offset;

    return gpu_params;
  }

 private:
  static GPU_LUT3D CreateLUTTextureObject(const std::vector<float>& lut_data, int edge_size) {
    GPU_LUT3D gpu_lut;
    gpu_lut.edge_size          = edge_size;

    const size_t        voxels = static_cast<size_t>(edge_size) * edge_size * edge_size;
    std::vector<float4> packed(voxels);
    for (size_t i = 0; i < voxels; ++i) {
      packed[i] = make_float4(lut_data[i * 3 + 0], lut_data[i * 3 + 1], lut_data[i * 3 + 2], 1.0f);
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaExtent            extent       = make_cudaExtent(edge_size, edge_size, edge_size);
    cudaMalloc3DArray(&gpu_lut.array, &channel_desc, extent);

    cudaMemcpy3DParms copy_params = {0};
    copy_params.srcPtr =
        make_cudaPitchedPtr((void*)packed.data(), edge_size * sizeof(float4), edge_size, edge_size);
    copy_params.dstArray = gpu_lut.array;
    copy_params.extent   = extent;
    copy_params.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy_params);

    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = gpu_lut.array;

    cudaTextureDesc tex_desc  = {};
    tex_desc.normalizedCoords = 1;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;

    cudaCreateTextureObject(&gpu_lut.texture_object, &res_desc, &tex_desc, nullptr);
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
      return CreateLUTTextureObject(lut.lut3d, lut.edge3d);
    }

    // TODO: Add support for 1D LUTs if needed
    throw std::runtime_error("GPUParamsConverter: Only 3D LUTs are supported for GPU processing.");
  };

  static GPU_LUT3D CreateLUTTextureObject(std::filesystem::path& path) {
    CubeLut lut;
    ParseCubeFile(path, lut);
    if (lut.Has3D()) {
      return CreateLUTTextureObject(lut.lut3d, lut.edge3d);
    }
    // TODO: Add support for 1D LUTs if needed
    throw std::runtime_error("GPUParamsConverter: Only 3D LUTs are supported for GPU processing.");
  };
};
};  // namespace puerhlab