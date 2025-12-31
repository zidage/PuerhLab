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

#include "edit/operators/color/saturation_op.hpp"

#include <memory>
#include <opencv2/core/mat.hpp>
#include <utility>

#include "edit/operators/color/conversion/Oklab_cvt.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {

SaturationOp::SaturationOp() : _saturation_offset(0) { ComputeScale(); }

SaturationOp::SaturationOp(float saturation_offset) : _saturation_offset(saturation_offset) {
  ComputeScale();
}

SaturationOp::SaturationOp(const nlohmann::json& params) { SetParams(params); }

/**
 * @brief Compute the scale from the offset
 *
 */
void SaturationOp::ComputeScale() {
  if (_saturation_offset >= 0.0f) {
    _scale = 1.0f + _saturation_offset / 100.0f;
  } else {
    _scale = 1.0f + _saturation_offset / 100.0f;
  }
}

void SaturationOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    OklabCvt::Oklab oklab_vec = OklabCvt::LinearRGB2Oklab(pixel);

    // Chroma = a^2 + b^2
    oklab_vec.a *= _scale;
    oklab_vec.b *= _scale;

    pixel = OklabCvt::Oklab2LinearRGB(oklab_vec);
  });
}



auto SaturationOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[_script_name] = _saturation_offset;

  return o;
}

void SaturationOp::SetParams(const nlohmann::json& params) {
  if (params.contains(_script_name)) {
    _saturation_offset = params[_script_name];
  } else {
    _saturation_offset = 0.0f;
  }
  ComputeScale();
}

void SaturationOp::SetGlobalParams(OperatorParams& params) const {
  params.saturation_offset = _scale;
}
};  // namespace puerhlab