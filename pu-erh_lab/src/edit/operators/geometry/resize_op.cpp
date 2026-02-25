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

#include "edit/operators/geometry/resize_op.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"
#ifdef HAVE_CUDA
#include "edit/operators/geometry/cuda_geometry_ops.hpp"
#endif


namespace puerhlab {
ResizeOp::ResizeOp(const nlohmann::json& params) { SetParams(params); }

void ResizeOp::Apply(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetCPUData();
  int   w   = img.cols;
  int   h   = img.rows;
  if (w <= 0 || h <= 0) {
    return;
  }

  if (!enable_scale_ && !enable_roi_) {
    return;
  }

  const float full_scale =
      enable_scale_
          ? std::min(1.0f,
                     static_cast<float>(maximum_edge_) / static_cast<float>(std::max(w, h)))
          : 1.0f;

  if (!enable_roi_ && full_scale >= (1.0f - 1e-4f)) {
    return;
  }

  if (enable_roi_) {
    const float roi_factor_x =
        std::clamp(roi_.resize_factor_x_ > 0.0f ? roi_.resize_factor_x_ : roi_.resize_factor_, 1e-4f,
                   1.0f);
    const float roi_factor_y =
        std::clamp(roi_.resize_factor_y_ > 0.0f ? roi_.resize_factor_y_ : roi_.resize_factor_, 1e-4f,
                   1.0f);
    int roi_w =
        std::clamp(static_cast<int>(std::lround(static_cast<float>(w) * roi_factor_x)), 1, w);
    int roi_h =
        std::clamp(static_cast<int>(std::lround(static_cast<float>(h) * roi_factor_y)), 1, h);
    int roi_x = std::clamp(roi_.x_, 0, std::max(0, w - roi_w));
    int roi_y = std::clamp(roi_.y_, 0, std::max(0, h - roi_h));

    cv::Rect roi_rect(roi_x, roi_y, roi_w, roi_h);
    cv::Mat  roi_img = img(roi_rect).clone();

    const float roi_scale =
        std::min(1.0f,
                 static_cast<float>(maximum_edge_) / static_cast<float>(std::max(roi_w, roi_h)));
    if (roi_scale < (1.0f - 1e-4f)) {
      const int out_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(roi_w) * roi_scale)));
      const int out_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(roi_h) * roi_scale)));
      cv::resize(roi_img, roi_img, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
    }
    img = roi_img;
    return;
  }

  const int out_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * full_scale)));
  const int out_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * full_scale)));
  cv::resize(img, img, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
}

void ResizeOp::ApplyGPU(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetGPUData();
  int   w   = img.cols;
  int   h   = img.rows;
  if (w <= 0 || h <= 0) return;
  if (!enable_scale_ && !enable_roi_) return;

  const float full_scale =
      enable_scale_
          ? std::min(1.0f,
                     static_cast<float>(maximum_edge_) / static_cast<float>(std::max(w, h)))
          : 1.0f;

  if (!enable_roi_ && full_scale >= (1.0f - 1e-4f)) {
    return;
  }

  if (enable_roi_) {
    const float roi_factor_x =
        std::clamp(roi_.resize_factor_x_ > 0.0f ? roi_.resize_factor_x_ : roi_.resize_factor_, 1e-4f,
                   1.0f);
    const float roi_factor_y =
        std::clamp(roi_.resize_factor_y_ > 0.0f ? roi_.resize_factor_y_ : roi_.resize_factor_, 1e-4f,
                   1.0f);
    int roi_w =
        std::clamp(static_cast<int>(std::lround(static_cast<float>(w) * roi_factor_x)), 1, w);
    int roi_h =
        std::clamp(static_cast<int>(std::lround(static_cast<float>(h) * roi_factor_y)), 1, h);
    int roi_x = std::clamp(roi_.x_, 0, std::max(0, w - roi_w));
    int roi_y = std::clamp(roi_.y_, 0, std::max(0, h - roi_h));
    cv::Rect roi_rect(roi_x, roi_y, roi_w, roi_h);

    cv::cuda::GpuMat roi_src = img(roi_rect).clone();

    const float roi_scale =
        std::min(1.0f,
                 static_cast<float>(maximum_edge_) / static_cast<float>(std::max(roi_w, roi_h)));
    if (roi_scale >= (1.0f - 1e-4f)) {
      img = roi_src;
      return;
    }

    const int out_w =
        std::max(1, static_cast<int>(std::lround(static_cast<float>(roi_w) * roi_scale)));
    const int out_h =
        std::max(1, static_cast<int>(std::lround(static_cast<float>(roi_h) * roi_scale)));

    cv::cuda::GpuMat roi_dst;
#ifdef HAVE_CUDA
    CUDA::ResizeAreaApprox(roi_src, roi_dst, cv::Size(out_w, out_h));
#else
    throw std::runtime_error("ResizeOp::ApplyGPU requires HAVE_CUDA");
#endif
    img = std::move(roi_dst);
    return;
  }

  const int out_w =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * full_scale)));
  const int out_h =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * full_scale)));

  cv::cuda::GpuMat dst;
#ifdef HAVE_CUDA
  CUDA::ResizeAreaApprox(img, dst, cv::Size(out_w, out_h));
#else
  throw std::runtime_error("ResizeOp::ApplyGPU requires HAVE_CUDA");
#endif
  img = std::move(dst);
}

auto ResizeOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  nlohmann::json inner;
  inner["enable_scale"] = enable_scale_;
  inner["maximum_edge"] = maximum_edge_;
  inner["enable_roi"]   = enable_roi_;
  inner["roi"]          = {{"x", roi_.x_},
                           {"y", roi_.y_},
                           {"resize_factor_x", roi_.resize_factor_x_},
                           {"resize_factor_y", roi_.resize_factor_y_},
                           {"resize_factor", roi_.resize_factor_}};

  params[script_name_]  = inner;
  return params;
}

auto ResizeOp::SetParams(const nlohmann::json& params) -> void {
  if (params.contains(script_name_)) {
    auto inner = params.at(script_name_);
    if (inner.contains("enable_scale")) {
      enable_scale_ = inner.at("enable_scale").get<bool>();
    } else {
      enable_scale_ = false;
    }
    if (inner.contains("maximum_edge")) {
      maximum_edge_ = inner.at("maximum_edge").get<int>();
    } else {
      maximum_edge_ = 2048;
    }
    if (inner.contains("enable_roi")) {
      enable_roi_ = inner.at("enable_roi").get<bool>();
    } else {
      enable_roi_ = false;
    }
    if (enable_roi_ && inner.contains("roi")) {
      auto roi_json       = inner.at("roi");
      roi_.x_             = roi_json.value("x", 0);
      roi_.y_             = roi_json.value("y", 0);
      roi_.resize_factor_ = roi_json.value("resize_factor", 1.0f);
      roi_.resize_factor_x_ = roi_json.value("resize_factor_x", roi_.resize_factor_);
      roi_.resize_factor_y_ = roi_json.value("resize_factor_y", roi_.resize_factor_);
    }
  } else {
    enable_scale_ = false;
    maximum_edge_ = 2048;
    enable_roi_   = false;
    roi_          = {0, 0, 1.0f, 1.0f, 1.0f};
  }
}

void ResizeOp::SetGlobalParams(OperatorParams&) const {
  // throw std::runtime_error("ResizeOp does not support global parameters.");
  // DO NOTHING
}

void ResizeOp::EnableGlobalParams(OperatorParams&, bool) {
  // Still DO NOTHING
  // ResizeOp is not a streamable operator
}
};  // namespace puerhlab
