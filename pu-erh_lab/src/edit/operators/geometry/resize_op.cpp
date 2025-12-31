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
  if (std::max(w, h) <= _maximum_edge) {
    return;
  }

  float   scale = static_cast<float>(_maximum_edge) / static_cast<float>(std::max(w, h));
  cv::Mat roi_img;
  if (enable_roi) {
    int roi_w = static_cast<int>(w * roi.resize_factor);
    int roi_h = static_cast<int>(h * roi.resize_factor);
    roi_w     = std::min(roi_w, w - roi.x);
    roi_h     = std::min(roi_h, h - roi.y);
    cv::Rect roi_rect(roi.x, roi.y, roi_w, roi_h);
    roi_img     = img(roi_rect);
    float scale = static_cast<float>(_maximum_edge) / static_cast<float>(std::max(roi_w, roi_h));
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
  inner["maximum_edge"] = _maximum_edge;
  inner["enable_roi"]   = enable_roi;
  inner["roi"]          = {{"x", roi.x}, {"y", roi.y}, {"resize_factor", roi.resize_factor}};

  params[_script_name]  = inner;
  return params;
}

auto ResizeOp::SetParams(const nlohmann::json& params) -> void {
  if (params.contains(_script_name)) {
    auto inner = params.at(_script_name);
    if (inner.contains("maximum_edge")) {
      _maximum_edge = inner.at("maximum_edge").get<int>();
    } else {
      _maximum_edge = 2048;
    }
    if (inner.contains("enable_roi")) {
      enable_roi = inner.at("enable_roi").get<bool>();
    } else {
      enable_roi = false;
    }
    if (enable_roi && inner.contains("roi")) {
      auto roi_json     = inner.at("roi");
      roi.x             = roi_json.value("x", 0);
      roi.y             = roi_json.value("y", 0);
      roi.resize_factor = roi_json.value("resize_factor", 1.0f);
    }
  } else {
    _maximum_edge = 2048;
  }
}

void ResizeOp::SetGlobalParams(OperatorParams&) const {
  throw std::runtime_error("ResizeOp does not support global parameters.");
}
};  // namespace puerhlab