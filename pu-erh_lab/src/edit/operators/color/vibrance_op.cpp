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

#include "edit/operators/color/vibrance_op.hpp"

#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <utility>

#include "edit/operators/color/conversion/Oklab_cvt.hpp"
#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {

VibranceOp::VibranceOp() : vibrance_offset_(0) {}

VibranceOp::VibranceOp(float vibrance_offset) : vibrance_offset_(vibrance_offset) {}

VibranceOp::VibranceOp(const nlohmann::json& params) { SetParams(params); }

auto VibranceOp::ComputeScale(float chroma) -> float {
  // chroma in [0, max], vibrance_offset in [-100, 100]
  float strength = vibrance_offset_ / 100.0f;

  // Protect already highly saturated color
  float falloff  = std::exp(-3.0f * chroma);

  return 1.0f + strength * falloff;
}

void VibranceOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    // Adpated from https://github.com/tannerhelland/PhotoDemon
    float r = pixel[0], g = pixel[1], b = pixel[2];

    float max_val = std::max({r, g, b});
    float min_val = std::min({r, g, b});
    float chroma  = max_val - min_val;

    float scale   = ComputeScale(chroma);

    if (vibrance_offset_ >= 0.0f) {
      float luma = r * 0.299f + g * 0.587f + b * 0.114f;

      r          = luma + (r - luma) * scale;
      g          = luma + (g - luma) * scale;
      b          = luma + (b - luma) * scale;

    } else {
      float avg = (r + g + b) / 3.0f;
      r += (avg - r) * (1.0f - scale);
      g += (avg - g) * (1.0f - scale);
      b += (avg - b) * (1.0f - scale);
    }

    // clamp
    r     = std::clamp(r, 0.0f, 1.0f);
    g     = std::clamp(g, 0.0f, 1.0f);
    b     = std::clamp(b, 0.0f, 1.0f);

    pixel = {r, g, b};
  });
}


auto VibranceOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[script_name_] = vibrance_offset_;

  return o;
}

void VibranceOp::SetParams(const nlohmann::json& params) {
  if (params.contains(script_name_)) {
    vibrance_offset_ = params[script_name_].get<float>() / 100.0f;
  } else {
    vibrance_offset_ = 0.0f;
  }
}

void VibranceOp::SetGlobalParams(OperatorParams& params) const {
  params.vibrance_offset_ = vibrance_offset_;
}

void VibranceOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.vibrance_enabled_ = enable;
}
};  // namespace puerhlab