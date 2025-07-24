#include "edit/operators/color/vibrance_op.hpp"

#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <utility>

#include "edit/operators/color/conversion/Oklab_cvt.hpp"
#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {
VibranceOpRegister::VibranceOpRegister() {
  OperatorFactory::Instance().Register(OperatorType::VIBRANCE, [](const nlohmann::json& params) {
    return std::make_shared<VibranceOp>(params);
  });
}

static VibranceOpRegister vibrance_op_reg;

VibranceOp::VibranceOp() : _vibrance_offset(0) {}

VibranceOp::VibranceOp(float vibrance_offset) : _vibrance_offset(vibrance_offset) {}

VibranceOp::VibranceOp(const nlohmann::json& params) { SetParams(params); }

auto VibranceOp::ComputeScale(float chroma) -> float {
  // chroma in [0, max], vibrance_offset in [-100, 100]
  float strength = _vibrance_offset / 100.0f;

  // Protect already highly saturated color
  float falloff  = std::exp(-3.0f * chroma);

  return 1.0f + strength * falloff;
}

auto VibranceOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& img = input.GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    // Adpated from https://github.com/tannerhelland/PhotoDemon
    float r = pixel[0], g = pixel[1], b = pixel[2];

    float avg      = (r + g + g + b) / 4.0f;
    float max_val  = std::max({r, g, b});

    // vibrance_bias ∈ [-100, 100]
    float strength = -_vibrance_offset * 0.02f;

    float amt      = std::abs(max_val - avg) * strength;

    if (r != max_val) r += (max_val - r) * amt;
    if (g != max_val) g += (max_val - g) * amt;
    if (b != max_val) b += (max_val - b) * amt;

    // clamp
    r     = std::clamp(r, 0.0f, 1.0f);
    g     = std::clamp(g, 0.0f, 1.0f);
    b     = std::clamp(b, 0.0f, 1.0f);

    pixel = {r, g, b};
  });

  return {std::move(input)};
}

auto VibranceOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[_script_name] = _vibrance_offset;

  return o;
}

void VibranceOp::SetParams(const nlohmann::json& params) {
  if (params.contains(_script_name)) {
    _vibrance_offset = params[_script_name];
  } else {
    _vibrance_offset = 0.0f;
  }
}
};  // namespace puerhlab