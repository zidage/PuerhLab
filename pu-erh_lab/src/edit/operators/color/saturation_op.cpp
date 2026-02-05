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

SaturationOp::SaturationOp() : saturation_offset_(0) { ComputeScale(); }

SaturationOp::SaturationOp(float saturation_offset) : saturation_offset_(saturation_offset) {
  ComputeScale();
}

SaturationOp::SaturationOp(const nlohmann::json& params) { SetParams(params); }

/**
 * @brief Compute the scale from the offset
 *
 */
void SaturationOp::ComputeScale() {
  if (saturation_offset_ >= 0.0f) {
    scale_ = 1.0f + saturation_offset_ / 100.0f;
  } else {
    scale_ = 1.0f + saturation_offset_ / 100.0f;
  }
}

void SaturationOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    OklabCvt::Oklab oklab_vec = OklabCvt::LinearRGB2Oklab(pixel);

    // Chroma = a^2 + b^2
    oklab_vec.a_ *= scale_;
    oklab_vec.b_ *= scale_;

    pixel = OklabCvt::Oklab2LinearRGB(oklab_vec);
  });
}

void SaturationOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("SaturationOp: ApplyGPU not implemented");
}

auto SaturationOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[script_name_] = saturation_offset_;

  return o;
}

void SaturationOp::SetParams(const nlohmann::json& params) {
  if (params.contains(script_name_)) {
    saturation_offset_ = params[script_name_];
  } else {
    saturation_offset_ = 0.0f;
  }
  ComputeScale();
}

void SaturationOp::SetGlobalParams(OperatorParams& params) const {
  params.saturation_offset_ = scale_;
}

void SaturationOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.saturation_enabled_ = enable;
}
};  // namespace puerhlab