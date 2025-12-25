#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>

struct GPUOperatorParams {
  // Basic adjustment parameters
  bool                exposure_enabled       = true;
  float               exposure_offset        = 0.0f;

  bool                contrast_enabled       = true;
  float               contrast_scale         = 0.0f;

  // Shadows adjustment parameter
  bool                shadows_enabled        = true;
  float               shadows_offset         = 0.0f;
  float               shadows_x0             = 0.0f;
  float               shadows_x1             = 0.25f;
  float               shadows_y0             = 0.0f;
  float               shadows_y1             = 0.25f;
  float               shadows_m0             = 0.0f;
  float               shadows_m1             = 1.0f;
  float               shadows_dx             = 0.25f;

  // Highlights adjustment parameter
  bool                highlights_enabled     = true;
  const float         highlights_k           = 0.2f;
  float               highlights_offset      = 0.0f;
  const float         highlights_slope_range = 0.8f;
  float               highlights_m0          = 1.0f;
  float               highlights_m1          = 1.0f;
  float               highlights_x0          = 0.2f;
  float               highlights_y0          = 0.2f;
  float               highlights_y1          = 1.0f;
  float               highlights_dx          = 0.8f;

  // White and Black point adjustment parameters
  bool                white_enabled          = true;
  float               white_point            = 1.0f;

  bool                black_enabled          = true;
  float               black_point            = 0.0f;

  float               slope                  = 1.0f;
  // HLS adjustment parameters
  bool                hls_enabled            = true;
  float               target_hls[3]          = {0.0f, 0.0f, 0.0f};
  float               hls_adjustment[3]      = {0.0f, 0.0f, 0.0f};
  float               hue_range              = 0.0f;
  float               lightness_range        = 0.0f;
  float               saturation_range       = 0.0f;

  // Saturation adjustment parameter
  bool                saturation_enabled     = true;
  float               saturation_offset      = 0.0f;

  // Tint adjustment parameter
  bool                tint_enabled           = true;
  float               tint_offset            = 0.0f;

  // Vibrance adjustment parameter
  bool                vibrance_enabled       = true;
  float               vibrance_offset        = 0.0f;

  // Working space
  bool                is_working_space       = true;
  cudaTextureObject_t to_ws_lut              = 0;
  float               lut_max_coord_ws       = 1.0f;
  // TODO: NOT IMPLEMENTED

  // Look modification transform
  bool                lmt_enabled            = true;
  cudaTextureObject_t lmt_lut                = 0;
  float               lut_max_coord_lmt       = 1.0f;
  // TODO: NOT IMPLEMENTED

  // Output transform
  bool                to_output_enabled      = true;
  cudaTextureObject_t to_output_lut          = 0;
  float               lut_max_coord_output       = 1.0f;

  // Curve adjustment parameters
  bool                curve_enabled          = false;

  // Clarity adjustment parameter
  bool                clarity_enabled        = true;
  float               clarity_offset         = 0.0f;

  // Sharpen adjustment parameter
  bool                sharpen_enabled        = true;
  float               sharpen_offset         = 0.0f;
  float               sharpen_radius         = 3.0f;
  float               sharpen_threshold      = 0.0f;

  // Color wheel adjustment parameters
  bool                color_wheel_enabled    = true;
  float               lift_color_offset[3]   = {0.0f, 0.0f, 0.0f};
  float               lift_luminance_offset  = 0.0f;
  float               gamma_color_offset[3]  = {1.0f, 1.0f, 1.0f};
  float               gamma_luminance_offset = 0.0f;
  float               gain_color_offset[3]   = {1.0f, 1.0f, 1.0f};
  float               gain_luminance_offset  = 0.0f;
};