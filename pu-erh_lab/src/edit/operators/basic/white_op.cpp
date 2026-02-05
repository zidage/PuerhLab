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

#include "edit/operators/basic/white_op.hpp"

#include "edit/operators/op_base.hpp"
#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
WhiteOp::WhiteOp(float offset) : offset_(offset) {}

WhiteOp::WhiteOp(const nlohmann::json& params) { SetParams(params); }
auto WhiteOp::GetScale() -> float { return offset_ / 3.0f; }

void WhiteOp::Apply(std::shared_ptr<ImageBuffer> input) { (void)input; }

void WhiteOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("WhiteOp: ApplyGPU not implemented");
}

auto WhiteOp::GetParams() const -> nlohmann::json { return {script_name_, offset_}; }

void WhiteOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(script_name_)) {
    offset_      = 0.0f;
    y_intercept_ = 1.0f;
    black_point_ = 0.0f;
    slope_       = 1.0f;
  } else {
    offset_      = params[script_name_].get<float>() / 100.0f;
    y_intercept_ = 1.0f + offset_ / 3.0f;
    black_point_ = 0.0f;
    slope_       = (y_intercept_ - black_point_) / 1.0f;
  }
}

void WhiteOp::SetGlobalParams(OperatorParams& params) const {
  params.white_point_ = y_intercept_;
  params.slope_       = (params.white_point_ - params.black_point_);
}

void WhiteOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  if (enable) {
    params.white_point_ = y_intercept_;
    params.slope_       = (params.white_point_ - params.black_point_);
  } else {
    params.white_point_ = 1.0f;
    params.slope_       = 1.0f - params.black_point_;
  }
}
}  // namespace puerhlab