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

#include "edit/operators/color/HLS_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {

HLSOp::HLSOp()
    : target_hls_(0, 0.5f, 1.0f),
      hls_adjustment_(0.0f, 0.0f, 0.0f),
      hue_range_(15.0f),
      lightness_range_(0.1f),
      saturation_range_(0.1f) {}

HLSOp::HLSOp(const nlohmann::json& params) { SetParams(params); }

void HLSOp::SetTargetColor(const cv::Vec3f& bgr_color_normalized) {
  cv::Mat bgr_mat(1, 1, CV_32FC3);
  bgr_mat.at<cv::Vec3f>(0, 0) = bgr_color_normalized;

  cv::Mat HLS_mat;
  cv::cvtColor(bgr_mat, HLS_mat, cv::COLOR_BGR2HLS);
  target_hls_ = HLS_mat.at<cv::Vec3f>(0, 0);
}

void HLSOp::SetAdjustment(const cv::Vec3f& adjustment) { hls_adjustment_ = adjustment; }

void HLSOp::SetRanges(float h_range, float l_range, float s_range) {
  hue_range_        = h_range;
  lightness_range_  = l_range;
  saturation_range_ = s_range;
}

void HLSOp::Apply(std::shared_ptr<ImageBuffer> input) {
  if (cv::norm(hls_adjustment_, cv::NORM_L2SQR) < 1e-10) {
    return;
  }

  cv::Mat& img = input->GetCPUData();
  cv::Mat  HLS_img;
  cv::cvtColor(img, HLS_img, cv::COLOR_RGB2HLS);

  cv::Mat     mask     = cv::Mat::zeros(img.size(), CV_32F);
  const float target_h = target_hls_[0];
  const float target_l = target_hls_[1];
  const float target_s = target_hls_[2];

  HLS_img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* position) -> void {
    const float h                = pixel[0];
    const float l                = pixel[1];
    const float s                = pixel[2];

    float       hue_diff         = std::abs(h - target_h);
    float       hue_dist         = std::min(hue_diff, 360.0f - hue_diff);

    float       hue_weight       = std::max(0.0f, 1.0f - hue_dist / hue_range_);
    float       lightness_weight = std::max(0.0f, 1.0f - std::abs(l - target_l) / lightness_range_);
    float saturation_weight = std::max(0.0f, 1.0f - std::abs(s - target_s) / saturation_range_);
    mask.at<float>(position[0], position[1]) = hue_weight * lightness_weight * saturation_weight;
  });

  cv::Mat              adj_h = cv::Mat(img.size(), CV_32F, hls_adjustment_[0]);
  cv::Mat              adj_l = cv::Mat(img.size(), CV_32F, hls_adjustment_[1]);
  cv::Mat              adj_s = cv::Mat(img.size(), CV_32F, hls_adjustment_[2]);

  std::vector<cv::Mat> HLS_channels(3);
  cv::split(HLS_img, HLS_channels);

  cv::Mat hue_adjusted = HLS_channels[0] + adj_h.mul(mask);
  hue_adjusted.forEach<float>([](float& p, const int*) -> void {
    p = std::fmod(p, 360.0f);
    if (p < 0) p += 360.0f;
  });
  HLS_channels[0] = hue_adjusted;

  HLS_channels[1] += adj_l.mul(mask);
  HLS_channels[2] += adj_s.mul(mask);
  cv::threshold(HLS_channels[1], HLS_channels[1], 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(HLS_channels[2], HLS_channels[2], 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(HLS_channels[1], HLS_channels[1], 0.0f, 0.0f, cv::THRESH_TOZERO);
  cv::threshold(HLS_channels[2], HLS_channels[2], 0.0f, 0.0f, cv::THRESH_TOZERO);

  cv::merge(HLS_channels, img);
  cv::cvtColor(img, img, cv::COLOR_HLS2RGB);

  cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
}


auto HLSOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["target_hls"] = std::array<float, 3>{target_hls_[0], target_hls_[1], target_hls_[2]};
  inner["hls_adj"] =
      std::array<float, 3>{hls_adjustment_[0], hls_adjustment_[1], hls_adjustment_[2]};
  inner["h_range"] = hue_range_;
  inner["l_range"] = lightness_range_;
  inner["s_range"] = saturation_range_;

  o[script_name_]  = inner;
  return o;
}

void HLSOp::SetParams(const nlohmann::json& params) {
  if (params.contains(script_name_)) {
    nlohmann::json inner = params[script_name_];
    if (inner.contains("target_hls")) {
      auto tgt_hls = inner["target_hls"].get<std::array<float, 3>>();
      target_hls_  = {tgt_hls[0], tgt_hls[1], tgt_hls[2]};
    }
    if (inner.contains("hls_adj")) {
      auto hls_adj    = inner["hls_adj"].get<std::array<float, 3>>();
      hls_adjustment_ = {hls_adj[0], hls_adj[1], hls_adj[2]};
    }
    if (inner.contains("h_range")) {
      hue_range_ = inner["h_range"].get<float>();
    }
    if (inner.contains("l_range")) {
      lightness_range_ = inner["l_range"].get<float>();
    }
    if (inner.contains("s_range")) {
      saturation_range_ = inner["s_range"].get<float>();
    }
  }
}

void HLSOp::SetGlobalParams(OperatorParams& params) const {
  // No global params to set for HLSOp
  params.target_hls_[0]     = target_hls_[0];
  params.target_hls_[1]     = target_hls_[1];
  params.target_hls_[2]     = target_hls_[2];

  params.hls_adjustment_[0] = hls_adjustment_[0];
  ;
  params.hls_adjustment_[1] = hls_adjustment_[1];
  params.hls_adjustment_[2] = hls_adjustment_[2];

  params.hue_range_         = hue_range_;
  params.lightness_range_   = lightness_range_;
  params.saturation_range_  = saturation_range_;
}
};  // namespace puerhlab