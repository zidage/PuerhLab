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

#include "edit/operators/op_base.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
ResizeOp::ResizeOp(const nlohmann::json& params) { SetParams(params); }

void ResizeOp::Apply(std::shared_ptr<ImageBuffer> input) {
  auto& img = input->GetCPUData();
  int   w   = img.cols;
  int   h   = img.rows;
  if (std::max(w, h) <= maximum_edge_) {
    return;
  }

  if (!enable_scale_ && !enable_roi_) {
    return;
  }
  float scale =
      enable_scale_ ? static_cast<float>(maximum_edge_) / static_cast<float>(std::max(w, h)) : 1.0f;
  cv::Mat roi_img;
  if (enable_roi_) {
    int roi_w = static_cast<int>(w * roi_.resize_factor_);
    int roi_h = static_cast<int>(h * roi_.resize_factor_);
    roi_w     = std::min(roi_w, w - roi_.x_);
    roi_h     = std::min(roi_h, h - roi_.y_);
    cv::Rect roi_rect(roi_.x_, roi_.y_, roi_w, roi_h);
    roi_img     = img(roi_rect).clone();  // Clone to create contiguous memory
    float scale = static_cast<float>(maximum_edge_) / static_cast<float>(std::max(roi_w, roi_h));
    cv::resize(roi_img, roi_img,
               cv::Size(static_cast<int>(roi_w * scale), static_cast<int>(roi_h * scale)), 0, 0,
               cv::INTER_AREA);
    img = roi_img;
    return;
  }
  cv::resize(img, img, cv::Size(static_cast<int>(w * scale), static_cast<int>(h * scale)), 0, 0,
             cv::INTER_AREA);
}

auto ResizeOp::GetParams() const -> nlohmann::json {
  nlohmann::json params;
  nlohmann::json inner;
  inner["enable_scale"] = enable_scale_;
  inner["maximum_edge"] = maximum_edge_;
  inner["enable_roi"]   = enable_roi_;
  inner["roi"]          = {{"x", roi_.x_}, {"y", roi_.y_}, {"resize_factor", roi_.resize_factor_}};

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
    }
  } else {
    enable_scale_ = false;
    maximum_edge_ = 2048;
    enable_roi_   = false;
    roi_          = {0, 0, 1.0f};
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