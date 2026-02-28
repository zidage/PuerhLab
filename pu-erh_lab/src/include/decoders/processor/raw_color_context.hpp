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
};

}  // namespace puerhlab
