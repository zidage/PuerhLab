#include "edit/operators/basic/highlight_op.hpp"

#include <cmath>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
HighlightsOp::HighlightsOp(float offset) : _offset(offset) {
  _ctrl_param = hw::Mul(hw::Set(hw::ScalableTag<float>(), offset / 100.0f),
                        hw::Set(hw::ScalableTag<float>(), 40.0f));
}

HighlightsOp::HighlightsOp(const nlohmann::json& params) {
  SetParams(params);
  _ctrl_param = hw::Mul(hw::Set(hw::ScalableTag<float>(), _offset / 100.0f),
                        hw::Set(hw::ScalableTag<float>(), 40.0f));
}

auto HighlightsOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  auto scaled_luminance = hw::Div(luminance, hw::Set(hw::ScalableTag<float>(), 100.0f));
  auto t_square         = hw::Mul(scaled_luminance, scaled_luminance);

  auto result           = hw::MulAdd(_ctrl_param, t_square, luminance);
  return result;
}

static inline float h00(float t) { return 2 * t * t * t - 3 * t * t + 1; }
static inline float h10(float t) { return t * t * t - 2 * t * t + t; }
static inline float h01(float t) { return -2 * t * t * t + 3 * t * t; }
static inline float h11(float t) { return t * t * t - t * t; }

static inline float clampf(float v, float a, float b) { return std::max(a, std::min(b, v)); }

// Luminance (linear, BGR)
static inline float Luma(const cv::Vec3f& bgr) {
  return 0.0722f * bgr[2] + 0.7152f * bgr[1] + 0.2126f * bgr[0];
}

static inline float Luma(const Pixel& rgb) {
  return 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
}

auto HighlightsOp::GetOutput(cv::Vec3f& input) -> cv::Vec3f {
  // float scaled_luminance = luminance / 100.0f;
  // clamp parameters reasonably
  float       control     = clampf(_offset / 100.0f, -1.0f, 1.0f);
  float       knee_start  = clampf(0.2f, 0.0f, 1.0f);  // ensure <= whitepoint
  // map control -> slope at whitepoint (m1)
  // design: control = +1 => strong compression (m1 -> small, e.g. 0.2)
  //         control =  0 => identity slope (1.0)
  //         control = -1 => boost highlights (m1 -> >1, e.g. 1.8)
  const float slope_range = 0.8f;  // how far slope can move from 1.0 (tuneable)
  float       m1          = 1.0f - control * slope_range;  // in [1-slope_range, 1+slope_range]

  // endpoints for Hermite between x0 = knee_start, x1 = whitepoint
  float       x0          = knee_start;
  float       x1          = 1.0f;
  float       y0          = x0;  // keep continuity (identity at x0)
  float       y1          = x1;  // identity at x1 (we'll control slope to shape shoulder)

  // slope at x0: preserve continuity with identity -> m0 = 1
  float       m0          = 1.0f;

  // For Hermite formula we need derivatives dy/dx at endpoints.
  // m0 and m1 are dy/dx at x0 and x1 respectively.
  // But Hermite cubic uses tangents scaled by (x1-x0) in the basis:
  float       dx          = (x1 - x0);
  // handle degenerate case: if dx==0, do nothing
  if (dx <= 1e-6f) {
    // fallback: no curve region; just apply linear scaling beyond whitepoint
    float L = Luma(input);
    float outL;
    if (L <= x0)
      outL = L;
    else
      outL = y1 + (L - x1) * m1;  // linear extrapolate
    float scale = (L > 1e-8f) ? outL / L : 1.0f;
    return input * scale;
  }

  // compute input luminance
  float L    = Luma(input);
  float outL = L;

  if (L <= x0) {
    // below knee_start: identity
    outL = L;
  } else if (L < x1) {
    // inside the Hermite segment: parameterize t in [0,1]
    float t   = (L - x0) / dx;
    // Hermite interpolation:
    float H00 = h00(t);
    float H10 = h10(t);
    float H01 = h01(t);
    float H11 = h11(t);
    // note: tangents in Hermite are (dx * m0) and (dx * m1)
    outL      = H00 * y0 + H10 * (dx * m0) + H01 * y1 + H11 * (dx * m1);
  } else {
    // L >= whitepoint: linear extrapolate using slope m1
    outL = y1 + (L - x1) * m1;
  }

  // avoid negative or NaN
  if (!std::isfinite(outL)) outL = L;
  // Preserve hue/chroma by scaling RGB by ratio outL/L (guard L==0)
  float     scale = (L > 1e-8f) ? (outL / L) : 1.0f;
  cv::Vec3f out   = input * scale;
  // Do not clamp here — keep HDR for downstream processing. (Optionally clamp if you want)
  return out;
}

auto HighlightsOp::GetScale() -> float { return _offset / 300.0f; }

void HighlightsOp::Apply(std::shared_ptr<ImageBuffer> input) {
  ToneRegionOp<HighlightsOp>::Apply(input);
}

auto HighlightsOp::ToKernel() const -> Kernel {
  return Kernel{._type = Kernel::Type::Point,
                ._func = PointKernelFunc([c = _curve.control, k = _curve.knee_start, m0 = _curve.m0,
                                          m1 = _curve.m1, dx = _curve.dx](Pixel& in) {
                  float L    = 0.2126f * in.r + 0.7152f * in.g + 0.0722f * in.b;
                  float outL = L;

                  if (L <= k) {
                    // below knee_start: identity
                    outL = L;
                  } else if (L < 1.0f) {
                    // inside the Hermite segment: parameterize t in [0,1]
                    float t   = (L - k) / dx;
                    // Hermite interpolation:
                    float H00 = 2 * t * t * t - 3 * t * t + 1;
                    float H10 = t * t * t - 2 * t * t + t;
                    float H01 = -2 * t * t * t + 3 * t * t;
                    float H11 = t * t * t - t * t;
                    // note: tangents in Hermite are (dx * m0) and (dx * m1)
                    outL      = H00 * k + H10 * (dx * m0) + H01 * 1.0f + H11 * (dx * m1);
                  } else {
                    // L >= whitepoint: linear extrapolate using slope m1
                    outL = 1.0f + (L - 1.0f) * m1;
                  }

                  // avoid negative or NaN
                  if (!std::isfinite(outL)) outL = L;
                  // Preserve hue/chroma by scaling RGB by ratio outL/L (guard L==0)
                  float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
                  in.r *= scale;
                  in.g *= scale;
                  in.b *= scale;
                })};
}

auto HighlightsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void HighlightsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
  _curve.control    = clampf(_offset / 100.0f, -1.0f, 1.0f);
  _curve.knee_start = clampf(0.2f, 0.0f, 1.0f);  // ensure <= whitepoint
  // map control -> slope at whitepoint (m1)
  // design: control = +1 => strong compression (m1 -> small, e.g. 0.2)
  //         control =  0 => identity slope (1.0)
  //         control = -1 => boost highlights (m1 -> >1, e.g. 1.8)
  _curve.m1 = 1.0f - _curve.control * _curve.slope_range;  // in [1-slope_range, 1+slope_range]

  // endpoints for Hermite between x0 = knee_start, x1 = whitepoint
  _curve.x0 = _curve.knee_start;
  _curve.y0 = _curve.x0;  // keep continuity (identity at x0)
  _curve.y1 = _curve.x1;  // identity at x1 (we'll control slope to shape shoulder)

  // For Hermite formula we need derivatives dy/dx at endpoints.
  // m0 and m1 are dy/dx at x0 and x1 respectively.
  // But Hermite cubic uses tangents scaled by (x1-x0) in the basis:
  _curve.dx = (_curve.x1 - _curve.x0);
}
}  // namespace puerhlab