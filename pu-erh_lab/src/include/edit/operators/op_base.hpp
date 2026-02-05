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
#include <OpenColorIO/OpenColorTypes.h>

#include <memory>

#include "image/image_buffer.hpp"
#include "json.hpp"
#include "op_kernel.hpp"
#include "type/type.hpp"
#include "utils/color_utils.hpp"

namespace puerhlab {
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
  UNKNOWN  // For unrecognized operator types or placeholders
};

struct OperatorParams {
  // Basic adjustment parameters
  bool                         exposure_enabled_         = true;
  float                        exposure_offset_          = 0.0f;

  bool                         contrast_enabled_         = false;
  float                        contrast_scale_           = 0.0f;

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

  // White and Black point adjustment parameters
  bool                         white_enabled_            = true;
  float                        white_point_              = 1.0f;

  bool                         black_enabled_            = true;
  float                        black_point_              = 0.0f;

  float                        slope_                    = 1.0f;
  // HLS adjustment parameters
  bool                         hls_enabled_              = true;
  float                        target_hls_[3]            = {0.0f, 0.0f, 0.0f};
  float                        hls_adjustment_[3]        = {0.0f, 0.0f, 0.0f};
  float                        hue_range_                = 0.0f;
  float                        lightness_range_          = 0.0f;
  float                        saturation_range_         = 0.0f;

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

  // Look modification transform
  bool                         lmt_enabled_              = false;
  bool                         to_lmt_dirty_             = false;
  OCIO::ConstCPUProcessorRcPtr cpu_lmt_processor_        = nullptr;
  OCIO::ConstGPUProcessorRcPtr gpu_lmt_processor_        = nullptr;
  std::filesystem::path        lmt_lut_path_             = {};

  // To output space
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
  bool                         curve_enabled_            = false;
  std::vector<cv::Point2f>     curve_ctrl_pts_           = {};
  std::vector<float>           curve_h_                  = {};
  std::vector<float>           curve_m_                  = {};

  // Clarity adjustment parameter
  bool                         clarity_enabled_          = true;
  float                        clarity_offset_           = 0.0f;
  float                        clarity_radius_           = 5.0f;

  // Sharpen adjustment parameter
  bool                         sharpen_enabled_          = true;
  float                        sharpen_offset_           = 0.0f;
  float                        sharpen_radius_           = 3.0f;
  float                        sharpen_threshold_        = 0.0f;

  // Color wheel adjustment parameters
  bool                         color_wheel_enabled_      = true;
  float                        lift_color_offset_[3]     = {0.0f, 0.0f, 0.0f};
  float                        lift_luminance_offset_    = 0.0f;
  float                        gamma_color_offset_[3]    = {1.0f, 1.0f, 1.0f};
  float                        gamma_luminance_offset_   = 0.0f;
  float                        gain_color_offset_[3]     = {1.0f, 1.0f, 1.0f};
  float                        gain_luminance_offset_    = 0.0f;
};

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

};  // namespace puerhlab
