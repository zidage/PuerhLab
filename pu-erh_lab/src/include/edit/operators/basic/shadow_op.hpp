#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"
#include "tone_region_op.hpp"

namespace puerhlab {

struct ShadowCurveParams {
  float       control;             // [-1, 1]
  float       toe_end     = 0.25;  // end of toe region in [0,1], e.g. 0.25
  const float slope_range = 0.8f;

  // Hermite between x0=0 and x1=toe_end
  float       m0;         // slope at blackpoint (x0)
  float       m1 = 1.0f;  // slope at x1 to keep continuity

  float       x0 = 0.0f;
  float       x1;         // = toe_end
  float       y0 = 0.0f;  // identity at x0
  float       y1;         // = x1, identity at x1

  float       dx;  // x1 - x0
};
class ShadowsOp : public ToneRegionOp<ShadowsOp>, public OperatorBase<ShadowsOp> {
 private:
  float                                 _offset;
  ShadowCurveParams                     _curve{};

  hw::Vec<hw::ScalableTag<float>>       _scale;
  const hw::ScalableTag<float>          d;
  hw::Rebind<int32_t, decltype(d)>      di;
  const hw::Vec<hw::ScalableTag<float>> v_inv_max = hw::Set(d, 1.0f / 100.0f);
  ;  // 1.0f / 100.0f
  const hw::Vec<hw::ScalableTag<float>> v_pivot_scale =
      hw::Set(d, 1.0f / (1.0f - 0.05f));  // 1.0f / (1.0f - 0.05f)
  const hw::Vec<hw::ScalableTag<float>> v_pivot_offset =
      hw::Set(d, 0.05f / (1.0f - 0.05f));  // 0.05f / (1.0f - 0.05f)
  const hw::Vec<hw::ScalableTag<float>> v_zero          = hw::Set(d, 0.0f);
  const hw::Vec<hw::ScalableTag<float>> v_one           = hw::Set(d, 1.0f);
  const hw::Vec<hw::ScalableTag<float>> v_inv_threshold = hw::Set(d, 1.0f / 0.5f);
  const hw::Vec<hw::ScalableTag<float>> v_epsilon       = hw::Set(d, 1e-7f);

  static constexpr int                  kLutSize        = 1024;
  std::vector<float>                    _lut;
  const hw::Vec<hw::ScalableTag<float>> v_lut_scale = hw::Set(d, static_cast<float>(kLutSize - 1));

  const hw::Vec<hw::ScalableTag<float>> _pivot      = hw::Set(d, 0.05f);

  hw::Vec<hw::ScalableTag<float>>       v_gamma;
  float                                 _gamma;
  float                                 _inv_threshold = 1.0f / 0.5f;

  hw::Vec<hw::ScalableTag<float>>       _min           = hw::Set(hw::ScalableTag<float>(), 0.0f);
  hw::Vec<hw::ScalableTag<float>>       _max           = hw::Set(hw::ScalableTag<float>(), 100.0f);

  void                                  InitializeLUT();

 public:
  auto GetOutput(hw::Vec<hw::ScalableTag<float>>&) -> hw::Vec<hw::ScalableTag<float>>;
  auto GetOutput(cv::Vec3f&) -> cv::Vec3f;
  auto GetScale() -> float;
  static constexpr PriorityLevel     _priority_level    = 1;
  static constexpr PipelineStageName _affiliation_stage = PipelineStageName::Basic_Adjustment;
  static constexpr std::string_view  _canonical_name    = "Shadows";
  static constexpr std::string_view  _script_name       = "shadows";
  static constexpr ToneRegion        _tone_region       = ToneRegion::SHADOWS;

  ShadowsOp()                                           = default;
  ShadowsOp(float offset);
  ShadowsOp(const nlohmann::json& params);

  static void GetMask(cv::Mat& src, cv::Mat& mask);

  void        Apply(std::shared_ptr<ImageBuffer> input) override;
  auto        ToKernel() const -> Kernel override;
  auto        GetParams() const -> nlohmann::json override;
  void        SetParams(const nlohmann::json& params) override;
};
}  // namespace puerhlab