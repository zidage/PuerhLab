//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <string>

namespace puerhlab {

/// Lightweight struct carrying raw-file colour and lens metadata extracted
/// during decode or import.  Factored out of raw_processor.hpp so that
/// headers without heavyweight dependencies (Image, OperatorParams, …) can
/// use it without pulling in libraw / decoder_scheduler.
struct RawRuntimeColorContext {
  bool              valid_                   = false;
  bool              output_in_camera_space_  = false;
  float             cam_mul_[3]              = {1.0f, 1.0f, 1.0f};
  float             pre_mul_[3]              = {1.0f, 1.0f, 1.0f};
  float             cam_xyz_[9]              = {};
  float             rgb_cam_[9]              = {};
  std::string       camera_make_             = {};
  std::string       camera_model_            = {};
  bool              lens_metadata_valid_     = false;
  std::string       lens_make_               = {};
  std::string       lens_model_              = {};
  float             focal_length_mm_         = 0.0f;
  float             aperture_f_number_       = 0.0f;
  float             focus_distance_m_        = 0.0f;
  float             focal_35mm_mm_           = 0.0f;
  float             crop_factor_hint_        = 0.0f;

  // Adobe DNG colour matrices resolved at import time from the camera
  // matrix database.  Avoids repeated database lookups per frame.
  bool              color_matrices_valid_    = false;
  double            color_matrix_1_[9]       = {};
  double            color_matrix_2_[9]       = {};
};

}  // namespace puerhlab
