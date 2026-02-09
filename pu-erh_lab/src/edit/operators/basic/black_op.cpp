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

#include "edit/operators/basic/black_op.hpp"

#include <opencv2/core.hpp>
#include <string>

#include "image/image_buffer.hpp"

namespace puerhlab {
BlackOp::BlackOp(float offset) : offset_(offset) {}

BlackOp::BlackOp(const nlohmann::json& params) { SetParams(params); }

auto BlackOp::GetScale() -> float { return offset_ / 3.0f; }

void BlackOp::Apply(std::shared_ptr<ImageBuffer> input) { (void)input; }

void BlackOp::ApplyGPU(std::shared_ptr<ImageBuffer>) { throw std::runtime_error("BlackOp: ApplyGPU not implemented"); }

auto BlackOp::GetParams() const -> nlohmann::json {
  return {{std::string(script_name_), offset_ * 100.0f}};
}

void BlackOp::SetParams(const nlohmann::json& params) {
  float value = 0.0f;
  bool  found = false;

  if (params.is_object() && params.contains(script_name_)) {
    value = params[script_name_].get<float>();
    found = true;
  } else if (params.is_array() && params.size() == 2) {
    // Backward compatibility for legacy snapshots serialized as ["black", value].
    try {
      if (params[0].is_string() && params[0].get<std::string>() == script_name_) {
        value = params[1].get<float>();
        found = true;
      }
    } catch (...) {
    }
  }

  if (!found) {
    offset_      = 0.0f;
    y_intercept_ = 0.0f;
    slope_       = 1.0f;

  } else {
    offset_      = value / 100.0f;
    y_intercept_ = offset_ / 10.f;
    slope_       = (1.0f - y_intercept_) / 1.0f;
  }
}

void BlackOp::SetGlobalParams(OperatorParams& params) const {
  // Should only be called once SetParams has been called
  params.black_point_ = y_intercept_;
  params.slope_       = (params.white_point_ - params.black_point_);
}

void BlackOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  if (enable) {
    params.black_point_ = y_intercept_;
    params.slope_       = (params.white_point_ - params.black_point_);
  } else {
    params.black_point_ = 0.0f;
    params.slope_       = params.white_point_;
  }
}
}  // namespace puerhlab
