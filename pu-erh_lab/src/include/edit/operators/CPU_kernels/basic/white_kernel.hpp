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

#pragma once
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct WhiteOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.white_enabled_) return;
    p.r_ = (p.r_) * params.slope_ + params.black_point_;
    p.g_ = (p.g_) * params.slope_ + params.black_point_;
    p.b_ = (p.b_) * params.slope_ + params.black_point_;
  }
};
}  // namespace puerhlab