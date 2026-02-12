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
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
namespace {
constexpr int kHlsProfileCount = OperatorParams::kHlsProfileCount;
constexpr std::array<float, kHlsProfileCount> kDefaultHueProfiles = {
    0.0f, 45.0f, 90.0f, 135.0f, 180.0f, 225.0f, 270.0f, 315.0f};

auto WrapHueDegrees(float hue) -> float {
  hue = std::fmod(hue, 360.0f);
  if (hue < 0.0f) {
    hue += 360.0f;
  }
  return hue;
}

auto HueDistanceDegrees(float a, float b) -> float {
  const float diff = std::abs(WrapHueDegrees(a) - WrapHueDegrees(b));
  return std::min(diff, 360.0f - diff);
}

auto ClosestHueProfileIdx(float hue, const std::array<float, kHlsProfileCount>& profiles) -> int {
  int   best_idx  = 0;
  float best_dist = HueDistanceDegrees(hue, profiles[0]);
  for (int i = 1; i < kHlsProfileCount; ++i) {
    const float dist = HueDistanceDegrees(hue, profiles[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best_idx  = i;
    }
  }
  return best_idx;
}
}  // namespace

HLSOp::HLSOp()
    : target_hls_(0, 0.5f, 1.0f),
      hls_adjustment_(0.0f, 0.0f, 0.0f),
      hue_range_(15.0f),
      lightness_range_(0.1f),
      saturation_range_(0.1f) {
  hue_profile_values_ = kDefaultHueProfiles;
  hls_adjustment_table_.fill(cv::Vec3f(0.0f, 0.0f, 0.0f));
  hue_range_table_.fill(15.0f);
  active_profile_idx_ = 0;
}

HLSOp::HLSOp(const nlohmann::json& params) { SetParams(params); }

void HLSOp::SetTargetColor(const cv::Vec3f& bgr_color_normalized) {
  cv::Mat bgr_mat(1, 1, CV_32FC3);
  bgr_mat.at<cv::Vec3f>(0, 0) = bgr_color_normalized;

  cv::Mat HLS_mat;
  cv::cvtColor(bgr_mat, HLS_mat, cv::COLOR_BGR2HLS);
  target_hls_ = HLS_mat.at<cv::Vec3f>(0, 0);
  active_profile_idx_ = ClosestHueProfileIdx(target_hls_[0], hue_profile_values_);
  target_hls_[0]      = hue_profile_values_[active_profile_idx_];
  hls_adjustment_     = hls_adjustment_table_[active_profile_idx_];
  hue_range_          = hue_range_table_[active_profile_idx_];
}

void HLSOp::SetAdjustment(const cv::Vec3f& adjustment) {
  hls_adjustment_ = adjustment;
  if (active_profile_idx_ >= 0 && active_profile_idx_ < kHlsProfileCount) {
    hls_adjustment_table_[active_profile_idx_] = adjustment;
  }
}

void HLSOp::SetRanges(float h_range, float l_range, float s_range) {
  hue_range_        = h_range;
  lightness_range_  = l_range;
  saturation_range_ = s_range;
  if (active_profile_idx_ >= 0 && active_profile_idx_ < kHlsProfileCount) {
    hue_range_table_[active_profile_idx_] = h_range;
  }
}

void HLSOp::Apply(std::shared_ptr<ImageBuffer> input) {
  bool has_any_adjustment = false;
  for (const auto& adj : hls_adjustment_table_) {
    if (cv::norm(adj, cv::NORM_L2SQR) >= 1e-10) {
      has_any_adjustment = true;
      break;
    }
  }
  if (!has_any_adjustment) {
    return;
  }

  cv::Mat& img = input->GetCPUData();
  cv::Mat  HLS_img;
  cv::cvtColor(img, HLS_img, cv::COLOR_RGB2HLS);

  cv::Mat mask = cv::Mat::zeros(img.size(), CV_32F);
  cv::Mat adj_h(img.size(), CV_32F, 0.0f);
  cv::Mat adj_l(img.size(), CV_32F, 0.0f);
  cv::Mat adj_s(img.size(), CV_32F, 0.0f);

  HLS_img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* position) -> void {
    const float h = WrapHueDegrees(pixel[0]);
    const int   idx = ClosestHueProfileIdx(h, hue_profile_values_);

    const float target_h       = WrapHueDegrees(hue_profile_values_[idx]);
    const float hue_diff       = std::abs(h - target_h);
    const float hue_dist       = std::min(hue_diff, 360.0f - hue_diff);
    const float safe_hue_range = std::max(hue_range_table_[idx], 1e-6f);
    const float hue_weight     = std::max(0.0f, 1.0f - hue_dist / safe_hue_range);

    mask.at<float>(position[0], position[1]) = hue_weight;
    adj_h.at<float>(position[0], position[1]) = hls_adjustment_table_[idx][0];
    adj_l.at<float>(position[0], position[1]) = hls_adjustment_table_[idx][1];
    adj_s.at<float>(position[0], position[1]) = hls_adjustment_table_[idx][2];
  });

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

void HLSOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("HLSOp: ApplyGPU not implemented");
}

auto HLSOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  nlohmann::json hue_bins     = nlohmann::json::array();
  nlohmann::json adj_table    = nlohmann::json::array();
  nlohmann::json h_range_table = nlohmann::json::array();
  for (int i = 0; i < kHlsProfileCount; ++i) {
    hue_bins.push_back(hue_profile_values_[i]);
    adj_table.push_back(
        std::array<float, 3>{hls_adjustment_table_[i][0], hls_adjustment_table_[i][1],
                             hls_adjustment_table_[i][2]});
    h_range_table.push_back(hue_range_table_[i]);
  }
  inner["hue_bins"]       = std::move(hue_bins);
  inner["hls_adj_table"]  = std::move(adj_table);
  inner["h_range_table"]  = std::move(h_range_table);

  const int active_idx = std::clamp(active_profile_idx_, 0, kHlsProfileCount - 1);
  inner["target_hls"]  = std::array<float, 3>{hue_profile_values_[active_idx], target_hls_[1],
                                               target_hls_[2]};
  inner["hls_adj"] =
      std::array<float, 3>{hls_adjustment_table_[active_idx][0], hls_adjustment_table_[active_idx][1],
                           hls_adjustment_table_[active_idx][2]};
  inner["h_range"] = hue_range_table_[active_idx];
  inner["l_range"] = lightness_range_;
  inner["s_range"] = saturation_range_;

  o[script_name_]  = inner;
  return o;
}

void HLSOp::SetParams(const nlohmann::json& params) {
  target_hls_        = {0.0f, 0.5f, 1.0f};
  hls_adjustment_    = {0.0f, 0.0f, 0.0f};
  hue_range_         = 15.0f;
  lightness_range_   = 0.1f;
  saturation_range_  = 0.1f;
  hue_profile_values_ = kDefaultHueProfiles;
  hls_adjustment_table_.fill(cv::Vec3f(0.0f, 0.0f, 0.0f));
  hue_range_table_.fill(15.0f);
  active_profile_idx_ = 0;

  if (!params.contains(script_name_)) {
    return;
  }

  nlohmann::json inner          = params[script_name_];
  bool           has_adj_table  = false;
  bool           has_range_table = false;

  if (inner.contains("hue_bins") && inner["hue_bins"].is_array()) {
    const auto& bins  = inner["hue_bins"];
    const int   count = std::min<int>(kHlsProfileCount, static_cast<int>(bins.size()));
    for (int i = 0; i < count; ++i) {
      try {
        hue_profile_values_[i] = WrapHueDegrees(bins[i].get<float>());
      } catch (...) {
      }
    }
  }

  if (inner.contains("hls_adj_table") && inner["hls_adj_table"].is_array()) {
    const auto& tbl  = inner["hls_adj_table"];
    const int   count = std::min<int>(kHlsProfileCount, static_cast<int>(tbl.size()));
    for (int i = 0; i < count; ++i) {
      try {
        if (tbl[i].is_array() && tbl[i].size() >= 3) {
          hls_adjustment_table_[i] =
              cv::Vec3f(tbl[i][0].get<float>(), tbl[i][1].get<float>(), tbl[i][2].get<float>());
          has_adj_table = true;
        }
      } catch (...) {
      }
    }
  }

  if (inner.contains("h_range_table") && inner["h_range_table"].is_array()) {
    const auto& tbl  = inner["h_range_table"];
    const int   count = std::min<int>(kHlsProfileCount, static_cast<int>(tbl.size()));
    for (int i = 0; i < count; ++i) {
      try {
        hue_range_table_[i] = std::max(tbl[i].get<float>(), 1e-6f);
        has_range_table     = true;
      } catch (...) {
      }
    }
  }

  if (inner.contains("target_hls")) {
    try {
      auto tgt_hls      = inner["target_hls"].get<std::array<float, 3>>();
      target_hls_       = {tgt_hls[0], tgt_hls[1], tgt_hls[2]};
      active_profile_idx_ = ClosestHueProfileIdx(target_hls_[0], hue_profile_values_);
      target_hls_[0]    = hue_profile_values_[active_profile_idx_];
    } catch (...) {
    }
  }
  if (inner.contains("hls_adj")) {
    try {
      auto hls_adj      = inner["hls_adj"].get<std::array<float, 3>>();
      hls_adjustment_   = {hls_adj[0], hls_adj[1], hls_adj[2]};
      if (!has_adj_table) {
        hls_adjustment_table_[active_profile_idx_] = hls_adjustment_;
      }
    } catch (...) {
    }
  }
  if (inner.contains("h_range")) {
    try {
      hue_range_        = inner["h_range"].get<float>();
      if (!has_range_table) {
        hue_range_table_[active_profile_idx_] = std::max(hue_range_, 1e-6f);
      }
    } catch (...) {
    }
  }
  if (inner.contains("l_range")) {
    try {
      lightness_range_ = inner["l_range"].get<float>();
    } catch (...) {
    }
  }
  if (inner.contains("s_range")) {
    try {
      saturation_range_ = inner["s_range"].get<float>();
    } catch (...) {
    }
  }

  active_profile_idx_ = std::clamp(active_profile_idx_, 0, kHlsProfileCount - 1);
  hls_adjustment_     = hls_adjustment_table_[active_profile_idx_];
  hue_range_          = hue_range_table_[active_profile_idx_];
  target_hls_[0]      = hue_profile_values_[active_profile_idx_];
}

void HLSOp::SetGlobalParams(OperatorParams& params) const {
  const int active_idx = std::clamp(active_profile_idx_, 0, kHlsProfileCount - 1);
  params.target_hls_[0]     = hue_profile_values_[active_idx];
  params.target_hls_[1]     = target_hls_[1];
  params.target_hls_[2]     = target_hls_[2];

  params.hls_adjustment_[0] = hls_adjustment_table_[active_idx][0];
  params.hls_adjustment_[1] = hls_adjustment_table_[active_idx][1];
  params.hls_adjustment_[2] = hls_adjustment_table_[active_idx][2];

  params.hue_range_         = hue_range_table_[active_idx];
  params.lightness_range_   = lightness_range_;
  params.saturation_range_  = saturation_range_;
  params.hls_profile_count_ = kHlsProfileCount;
  for (int i = 0; i < kHlsProfileCount; ++i) {
    params.hls_profile_hues_[i]            = hue_profile_values_[i];
    params.hls_profile_adjustments_[i][0]  = hls_adjustment_table_[i][0];
    params.hls_profile_adjustments_[i][1]  = hls_adjustment_table_[i][1];
    params.hls_profile_adjustments_[i][2]  = hls_adjustment_table_[i][2];
    params.hls_profile_hue_ranges_[i]      = hue_range_table_[i];
  }
}

void HLSOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.hls_enabled_ = enable;
}
};  // namespace puerhlab
