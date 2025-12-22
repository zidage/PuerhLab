#pragma once

#include <OpenColorIO/OpenColorTypes.h>

#include <memory>

#include "image/image_buffer.hpp"
#include "json.hpp"
#include "op_kernel.hpp"
#include "type/type.hpp"

namespace puerhlab {
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
  LMT,
  CLARITY,
  SHARPEN,
  COLOR_WHEEL,
  ACES_TONE_MAPPING,
  AUTO_EXPOSURE,
  UNKNOWN  // For unrecognized operator types or placeholders
};

struct OperatorParams {
  // Basic adjustment parameters
  bool                                     exposure_enabled       = true;
  float                                    exposure_offset        = 0.0f;
  float                                    contrast_scale         = 0.0f;

  // Shadows adjustment parameter
  bool                                     shadows_enabled        = true;
  float                                    shadows_offset         = 0.0f;
  float                                    shadows_x0             = 0.0f;
  float                                    shadows_x1             = 0.25f;
  float                                    shadows_y0             = 0.0f;
  float                                    shadows_y1             = 0.25f;
  float                                    shadows_m0             = 0.0f;
  float                                    shadows_m1             = 1.0f;
  float                                    shadows_dx             = 0.25f;

  // Highlights adjustment parameter
  bool                                     highlights_enabled     = true;
  const float                              highlights_k           = 0.2f;
  float                                    highlights_offset      = 0.0f;
  const float                              highlights_slope_range = 0.8f;
  float                                    highlights_m0          = 1.0f;
  float                                    highlights_m1          = 1.0f;
  float                                    highlights_x0          = 0.2f;
  float                                    highlights_y0          = 0.2f;
  float                                    highlights_y1          = 1.0f;
  float                                    highlights_dx          = 0.8f;

  // White and Black point adjustment parameters
  bool                                     white_black_enabled    = true;
  float                                    white_point            = 1.0f;
  float                                    white_slope            = 1.0f;

  bool                                     black_enabled          = true;
  float                                    black_slope            = 1.0f;
  float                                    black_point            = 0.0f;

  // HLS adjustment parameters
  bool                                     hls_enabled            = false;
  float                                    target_hls[3]          = {0.0f, 0.0f, 0.0f};
  float                                    hls_adjustment[3]      = {0.0f, 0.0f, 0.0f};
  float                                    hue_range              = 0.0f;
  float                                    lightness_range        = 0.0f;
  float                                    saturation_range       = 0.0f;

  // Saturation adjustment parameter
  bool                                     saturation_enabled     = true;
  float                                    saturation_offset      = 0.0f;

  // Tint adjustment parameter
  bool                                     tint_enabled           = true;
  float                                    tint_offset            = 0.0f;

  // Vibrance adjustment parameter
  bool                                     vibrance_enabled       = true;
  float                                    vibrance_offset        = 0.0f;

  // Working space
  bool                                     to_working_enabled     = true;
  OpenColorIO_v2_4::ConstCPUProcessorRcPtr to_working_processor   = nullptr;

  // Look modification transform
  bool                                     lmt_enabled            = false;
  OpenColorIO_v2_4::ConstCPUProcessorRcPtr lmt_processor          = nullptr;

  // To output space
  bool                                     to_output_enabled      = true;
  OpenColorIO_v2_4::ConstCPUProcessorRcPtr to_output_processor    = nullptr;

  // Curve adjustment parameters
  bool                                     curve_enabled          = false;
  std::vector<cv::Point2f>                 curve_ctrl_pts         = {};
  std::vector<float>                       curve_h                = {};
  std::vector<float>                       curve_m                = {};

  // Clarity adjustment parameter
  bool                                     clarity_enabled        = true;
  float                                    clarity_offset         = 0.0f;

  // Sharpen adjustment parameter
  bool                                     sharpen_enabled        = true;
  float                                    sharpen_offset         = 0.0f;
  float                                    sharpen_radius         = 3.0f;
  float                                    sharpen_threshold      = 0.0f;

  // Color wheel adjustment parameters
  bool                                     color_wheel_enabled    = false;
  float                                    lift_color_offset[3]   = {0.0f, 0.0f, 0.0f};
  float                                    lift_luminance_offset  = 0.0f;
  float                                    gamma_color_offset[3]  = {0.0f, 0.0f, 0.0f};
  float                                    gamma_luminance_offset = 0.0f;
  float                                    gain_color_offset[3]   = {0.0f, 0.0f, 0.0f};
  float                                    gain_luminance_offset  = 0.0f;
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

  virtual auto GetScriptName() const -> std::string          = 0;

  virtual auto GetPriorityLevel() const -> PriorityLevel     = 0;

  virtual auto GetStage() const -> PipelineStageName         = 0;

  virtual auto GetOperatorType() const -> OperatorType       = 0;

  virtual auto ToKernel() const -> Kernel                    = 0;

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
    return std::string(Derived::_canonical_name);
  }
  /**
   * @brief Get the script name of the operator (for JSON serialization)
   *
   * @return std::string
   */
  auto GetScriptName() const -> std::string override { return std::string(Derived::_script_name); }

  auto GetPriorityLevel() const -> PriorityLevel override { return Derived::_priority_level; }

  auto GetStage() const -> PipelineStageName override { return Derived::_affiliation_stage; }

  auto GetOperatorType() const -> OperatorType override { return Derived::_operator_type; }
};

struct OpStream {
  std::vector<std::shared_ptr<IOperatorBase>> _ops;

  bool AddToStream(const std::shared_ptr<IOperatorBase>& op) {
    // Ensure all kernels in the stream are of the same type
    _ops.push_back(op);
    return true;
  }

  void Clear() { _ops.clear(); }
};

struct PointOpTag {};
struct NeighborOpTag {};
};  // namespace puerhlab
