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

#include <libraw/libraw.h>

#include <opencv2/core.hpp>

#include "image/image_buffer.hpp"

namespace puerhlab {
namespace CUDA {
void ToLinearRef(cv::cuda::GpuMat& img, LibRaw& raw_processor);
};
};  // namespace puerhlab