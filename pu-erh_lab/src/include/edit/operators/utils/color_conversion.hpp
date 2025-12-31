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
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

namespace puerhlab {
inline static void GPUCvtColor(cv::Mat& src, cv::Mat& dst, int code) {
  cv::UMat uSrc, uDst;
  src.copyTo(uSrc);
  cv::cvtColor(uSrc, uDst, code);
  uDst.copyTo(dst);
}

inline static void GPUCvtColor(cv::UMat& src, cv::UMat& dst, int code) {
  cv::cvtColor(src, dst, code);
}
};  // namespace puerhlab