#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/utils/functions.hpp"
#include "hwy/contrib/math/math-inl.h"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {
  float normalized_offset = _offset / 100.0f;
  _gamma                  = std::pow(2.0f, -normalized_offset * 1.3f);
  v_gamma                 = hw::Set(d, _gamma);
}

ShadowsOp::ShadowsOp(const nlohmann::json& params) {
  SetParams(params);
  float normalized_offset = _offset / 100.0f;
  _gamma                  = std::pow(2.0f, -normalized_offset * 1.3f);
  v_gamma                 = hw::Set(d, _gamma);
}

auto ShadowsOp::GetScale() -> float { return _offset / 100.0f; }

void ShadowsOp::Apply(std::shared_ptr<ImageBuffer> input) { ToneRegionOp<ShadowsOp>::Apply(input); }

static inline float Luma(const Pixel& rgb) {
  return 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
}

auto ShadowsOp::ToKernel() const -> Kernel {
  return Kernel{
      ._type = Kernel::Type::Point,
      ._func = PointKernelFunc([&x0 = _curve.x0, &x1 = _curve.x1, &y0 = _curve.y0, &y1 = _curve.y1,
                                &m0 = _curve.m0, &m1 = _curve.m1, &dx = _curve.dx](Pixel& in) {
        const float eps = 1e-8f;
        float       L   = Luma(in);
        if (dx <= 1e-6f) return;

        float outL = L;
        if (L <= x0) {
          outL = 0.0f;
        } else if (L < x1) {
          float t   = (L - x0) / dx;
          float H00 = 2 * t * t * t - 3 * t * t + 1;
          float H10 = t * t * t - 2 * t * t + t;
          float H01 = -2 * t * t * t + 3 * t * t;
          float H11 = t * t * t - t * t;
          outL      = H00 * y0 + H10 * (dx * m0) + H01 * y1 + H11 * (dx * m1);
        } else {
          outL = L;
        }

        if (!std::isfinite(outL)) outL = L;
        float scale = (L > eps) ? (outL / L) : 1.0f;
        in.r *= scale;
        in.g *= scale;
        in.b *= scale;
      })};
}

auto ShadowsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void ShadowsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset        = params[_script_name].get<float>();
    _curve.control = std::clamp(_offset / 100.0f, -1.0f, 1.0f);
    _curve.toe_end = std::clamp(0.55f, 0.0f, 1.0f);
    _curve.m0      = 1.0f + _curve.control * _curve.slope_range;
    _curve.x1      = _curve.toe_end;
    _curve.y1      = _curve.x1;
    _curve.dx      = _curve.x1 - _curve.x0;
  }
}
}  // namespace puerhlab