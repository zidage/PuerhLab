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
WhiteOp::WhiteOp(float offset) : _offset(offset) {}

WhiteOp::WhiteOp(const nlohmann::json& params) { SetParams(params); }
auto WhiteOp::GetScale() -> float { return _offset / 3.0f; }

void WhiteOp::Apply(std::shared_ptr<ImageBuffer> input) { (void)input; }


auto WhiteOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void WhiteOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset      = 0.0f;
    _y_intercept = 1.0f;
    _black_point = 0.0f;
    _slope       = 1.0f;
  } else {
    _offset      = params[_script_name].get<float>() / 100.0f;
    _y_intercept = 1.0f + _offset / 3.0f;
    _black_point = 0.0f;
    _slope       = (_y_intercept - _black_point) / 1.0f;
  }
  _y_intercept_vec = simple_simd::set1_f32(_y_intercept);
  _black_point_vec = simple_simd::set1_f32(_black_point);
  _slope_vec       = simple_simd::set1_f32(_slope);
}

void WhiteOp::SetGlobalParams(OperatorParams& params) const {
  params.white_point = _y_intercept;
  params.slope       = (params.white_point - params.black_point);
}
}  // namespace puerhlab