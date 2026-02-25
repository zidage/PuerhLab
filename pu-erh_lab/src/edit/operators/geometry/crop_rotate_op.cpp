//  Copyright 2026 Yurun Zi
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

#include "edit/operators/geometry/crop_rotate_op.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>
#ifdef HAVE_CUDA
#include "edit/operators/geometry/cuda_geometry_ops.hpp"
#endif

namespace puerhlab {
namespace {
constexpr float kAngleEpsilon = 1e-4f;
constexpr float kCropEpsilon  = 1e-4f;
constexpr float kPi           = 3.14159265358979323846f;

auto ClampCropRect(NormalizedCropRect r) -> NormalizedCropRect {
  r.w_ = std::clamp(r.w_, kCropEpsilon, 1.0f);
  r.h_ = std::clamp(r.h_, kCropEpsilon, 1.0f);
  r.x_ = std::clamp(r.x_, 0.0f, 1.0f - r.w_);
  r.y_ = std::clamp(r.y_, 0.0f, 1.0f - r.h_);
  return r;
}

auto IsFullCropRect(const NormalizedCropRect& rect) -> bool {
  return std::abs(rect.x_) <= kCropEpsilon && std::abs(rect.y_) <= kCropEpsilon &&
         std::abs(rect.w_ - 1.0f) <= kCropEpsilon && std::abs(rect.h_ - 1.0f) <= kCropEpsilon;
}

auto NormalizeAngleDegrees(float angle_degrees) -> float {
  if (!std::isfinite(angle_degrees)) {
    return 0.0f;
  }
  angle_degrees = std::fmod(angle_degrees, 360.0f);
  if (angle_degrees > 180.0f) {
    angle_degrees -= 360.0f;
  } else if (angle_degrees < -180.0f) {
    angle_degrees += 360.0f;
  }
  return angle_degrees;
}

auto ClampCropRectForRotation(NormalizedCropRect rect, float angle_degrees) -> NormalizedCropRect {
  rect = ClampCropRect(rect);

  const float angle_radians = NormalizeAngleDegrees(angle_degrees) * (kPi / 180.0f);
  const float c             = std::abs(std::cos(angle_radians));
  const float s             = std::abs(std::sin(angle_radians));

  float cx = rect.x_ + rect.w_ * 0.5f;
  float cy = rect.y_ + rect.h_ * 0.5f;
  float hw = rect.w_ * 0.5f;
  float hh = rect.h_ * 0.5f;

  const float extent_x = (c * hw) + (s * hh);
  const float extent_y = (s * hw) + (c * hh);
  const float room_x   = std::max(0.0f, std::min(cx, 1.0f - cx));
  const float room_y   = std::max(0.0f, std::min(cy, 1.0f - cy));

  float scale = 1.0f;
  if (extent_x > room_x + kCropEpsilon) {
    scale = std::min(scale, room_x / std::max(extent_x, kCropEpsilon));
  }
  if (extent_y > room_y + kCropEpsilon) {
    scale = std::min(scale, room_y / std::max(extent_y, kCropEpsilon));
  }
  if (scale < 1.0f) {
    hw = std::max(kCropEpsilon * 0.5f, hw * scale);
    hh = std::max(kCropEpsilon * 0.5f, hh * scale);
  }

  rect.w_ = std::clamp(hw * 2.0f, kCropEpsilon, 1.0f);
  rect.h_ = std::clamp(hh * 2.0f, kCropEpsilon, 1.0f);
  rect.x_ = cx - rect.w_ * 0.5f;
  rect.y_ = cy - rect.h_ * 0.5f;
  return ClampCropRect(rect);
}

auto MakeWarpBorderScalar(int channels) -> cv::Scalar {
  if (channels == 4) {
    return cv::Scalar(0.0, 0.0, 0.0, 1.0);
  }
  return cv::Scalar::all(0.0);
}

auto ComputeCropRoi(int width, int height, const NormalizedCropRect& rect) -> cv::Rect {
  const auto clamped = ClampCropRect(rect);
  const int  roi_w =
      std::clamp(static_cast<int>(std::lround(static_cast<float>(width) * clamped.w_)), 1, width);
  const int roi_h =
      std::clamp(static_cast<int>(std::lround(static_cast<float>(height) * clamped.h_)), 1, height);
  int roi_x =
      std::clamp(static_cast<int>(std::lround(static_cast<float>(width) * clamped.x_)), 0, width - 1);
  int roi_y = std::clamp(static_cast<int>(std::lround(static_cast<float>(height) * clamped.y_)), 0,
                         height - 1);
  roi_x = std::clamp(roi_x, 0, std::max(0, width - roi_w));
  roi_y = std::clamp(roi_y, 0, std::max(0, height - roi_h));
  return cv::Rect(roi_x, roi_y, roi_w, roi_h);
}

auto BuildRotatedCropMatrix(int width, int height, const NormalizedCropRect& rect, float angle_degrees,
                            cv::Size& out_size) -> cv::Mat {
  const int out_w =
      std::clamp(static_cast<int>(std::lround(static_cast<float>(width) * rect.w_)), 1, width);
  const int out_h =
      std::clamp(static_cast<int>(std::lround(static_cast<float>(height) * rect.h_)), 1, height);

  const double angle_radians = static_cast<double>(angle_degrees) * (static_cast<double>(kPi) / 180.0);
  const double c             = std::cos(angle_radians);
  const double s             = std::sin(angle_radians);
  const double src_cx = static_cast<double>(rect.x_ + rect.w_ * 0.5f) * static_cast<double>(width);
  const double src_cy = static_cast<double>(rect.y_ + rect.h_ * 0.5f) * static_cast<double>(height);
  const double dst_cx = (static_cast<double>(out_w) - 1.0) * 0.5;
  const double dst_cy = (static_cast<double>(out_h) - 1.0) * 0.5;

  cv::Mat matrix(2, 3, CV_64F);
  matrix.at<double>(0, 0) = c;
  matrix.at<double>(0, 1) = -s;
  matrix.at<double>(1, 0) = s;
  matrix.at<double>(1, 1) = c;
  matrix.at<double>(0, 2) = src_cx - (c * dst_cx) + (s * dst_cy);
  matrix.at<double>(1, 2) = src_cy - (s * dst_cx) - (c * dst_cy);

  out_size = cv::Size(out_w, out_h);
  return matrix;
}
}  // namespace

CropRotateOp::CropRotateOp(const nlohmann::json& params) { SetParams(params); }

void CropRotateOp::Apply(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetCPUData();
  const int width  = img.cols;
  const int height = img.rows;
  if (width <= 0 || height <= 0) {
    return;
  }

  if (!enabled_ || !enable_crop_) {
    return;
  }

  const float angle_degrees = NormalizeAngleDegrees(angle_degrees_);
  const auto  crop_rect     = ClampCropRectForRotation(crop_rect_, angle_degrees);
  const bool  has_rotation  = std::abs(angle_degrees) > kAngleEpsilon;
  if (IsFullCropRect(crop_rect) && !has_rotation) {
    return;
  }

  if (!has_rotation) {
    const cv::Rect roi = ComputeCropRoi(width, height, crop_rect);
    img                = img(roi).clone();
    return;
  }

  // In rotated-crop-frame semantics, expand_to_fit is intentionally ignored.
  cv::Size out_size;
  cv::Mat  matrix = BuildRotatedCropMatrix(width, height, crop_rect, angle_degrees, out_size);
  cv::Mat  rotated_crop;
  cv::warpAffine(img, rotated_crop, matrix, out_size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, MakeWarpBorderScalar(img.channels()));
  img = rotated_crop;
}

void CropRotateOp::ApplyGPU(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetGPUData();
  const int width  = img.cols;
  const int height = img.rows;
  if (width <= 0 || height <= 0) {
    return;
  }

  if (!enabled_ || !enable_crop_) {
    return;
  }

  const float angle_degrees = NormalizeAngleDegrees(angle_degrees_);
  const auto  crop_rect     = ClampCropRectForRotation(crop_rect_, angle_degrees);
  const bool  has_rotation  = std::abs(angle_degrees) > kAngleEpsilon;
  if (IsFullCropRect(crop_rect) && !has_rotation) {
    return;
  }

  if (!has_rotation) {
    const cv::Rect roi = ComputeCropRoi(width, height, crop_rect);
    img                = img(roi).clone();
    return;
  }

  // In rotated-crop-frame semantics, expand_to_fit is intentionally ignored.
  cv::Size out_size;
  cv::Mat  matrix = BuildRotatedCropMatrix(width, height, crop_rect, angle_degrees, out_size);
  cv::cuda::GpuMat rotated_crop;
#ifdef HAVE_CUDA
  CUDA::WarpAffineLinear(img, rotated_crop, matrix, out_size, MakeWarpBorderScalar(img.channels()));
#else
  throw std::runtime_error("CropRotateOp::ApplyGPU requires HAVE_CUDA");
#endif
  img = std::move(rotated_crop);
}

auto CropRotateOp::GetParams() const -> nlohmann::json {
  return {{std::string(script_name_),
           {{"enabled", enabled_},
            {"angle_degrees", angle_degrees_},
            {"enable_crop", enable_crop_},
            {"crop_rect",
             {{"x", crop_rect_.x_}, {"y", crop_rect_.y_}, {"w", crop_rect_.w_}, {"h", crop_rect_.h_}}},
            {"expand_to_fit", expand_to_fit_}}}};
}

void CropRotateOp::SetParams(const nlohmann::json& params) {
  enabled_       = false;
  angle_degrees_ = 0.0f;
  enable_crop_   = false;
  crop_rect_     = {};
  expand_to_fit_ = true;

  if (!params.contains(script_name_) || !params[script_name_].is_object()) {
    return;
  }

  const auto& inner = params[script_name_];
  enabled_          = inner.value("enabled", enabled_);
  angle_degrees_    = inner.value("angle_degrees", angle_degrees_);
  enable_crop_      = inner.value("enable_crop", enable_crop_);
  expand_to_fit_    = inner.value("expand_to_fit", expand_to_fit_);
  if (inner.contains("crop_rect") && inner["crop_rect"].is_object()) {
    const auto& rect = inner["crop_rect"];
    crop_rect_.x_    = rect.value("x", crop_rect_.x_);
    crop_rect_.y_    = rect.value("y", crop_rect_.y_);
    crop_rect_.w_    = rect.value("w", crop_rect_.w_);
    crop_rect_.h_    = rect.value("h", crop_rect_.h_);
  }
  angle_degrees_ = NormalizeAngleDegrees(angle_degrees_);
  crop_rect_     = ClampCropRectForRotation(crop_rect_, angle_degrees_);
}

void CropRotateOp::SetGlobalParams(OperatorParams&) const {
  // Geometry operators are executed directly and do not populate stream global params.
}

void CropRotateOp::EnableGlobalParams(OperatorParams&, bool) {
  // Geometry operators are executed directly and do not populate stream global params.
}
}  // namespace puerhlab
