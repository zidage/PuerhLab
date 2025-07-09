#include "edit/operators/color/HLS_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
HLSOp::HLSOp()
    : _target_HLS(0, 0.5f, 1.0f),
      _HLS_adjustment(0.0f, 0.0f, 0.0f),
      _hue_range(15.0f),
      _lightness_range(0.1f),
      _saturation_range(0.1f) {}

void HLSOp::SetTargetColor(const cv::Vec3f& bgr_color_normalized) {
  cv::Mat bgr_mat(1, 1, CV_32FC3);
  bgr_mat.at<cv::Vec3f>(0, 0) = bgr_color_normalized;

  cv::Mat HLS_mat;
  cv::cvtColor(bgr_mat, HLS_mat, cv::COLOR_BGR2HLS);
  _target_HLS = HLS_mat.at<cv::Vec3f>(0, 0);
}

void HLSOp::SetAdjustment(const cv::Vec3f& adjustment) { _HLS_adjustment = adjustment; }

void HLSOp::SetRanges(float h_range, float l_range, float s_range) {
  _hue_range        = h_range;
  _lightness_range  = l_range;
  _saturation_range = s_range;
}

auto HLSOp::Apply(ImageBuffer& input) -> ImageBuffer {
  if (cv::norm(_HLS_adjustment, cv::NORM_L2SQR) < 1e-10) {
    return {std::move(input)};
  }

  cv::Mat& img = input.GetCPUData();
  cv::Mat  HLS_img;
  cv::cvtColor(img, HLS_img, cv::COLOR_BGR2HLS);

  cv::Mat     mask     = cv::Mat::zeros(img.size(), CV_32F);
  const float target_h = _target_HLS[0];
  const float target_l = _target_HLS[1];
  const float target_s = _target_HLS[2];

  HLS_img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* position) -> void {
    const float h                = pixel[0];
    const float l                = pixel[1];
    const float s                = pixel[2];

    float       hue_diff         = std::abs(h - target_h);
    float       hue_dist         = std::min(hue_diff, 360.0f - hue_diff);

    float       hue_weight       = std::max(0.0f, 1.0f - hue_dist / _hue_range);
    float       lightness_weight = std::max(0.0f, 1.0f - std::abs(l - target_l) / _lightness_range);
    float saturation_weight = std::max(0.0f, 1.0f - std::abs(s - target_s) / _saturation_range);
    mask.at<float>(position[0], position[1]) = hue_weight * lightness_weight * saturation_weight;
  });

  cv::Mat              adj_h = cv::Mat(img.size(), CV_32F, _HLS_adjustment[0]);
  cv::Mat              adj_l = cv::Mat(img.size(), CV_32F, _HLS_adjustment[1]);
  cv::Mat              adj_s = cv::Mat(img.size(), CV_32F, _HLS_adjustment[2]);

  std::vector<cv::Mat> HLS_channels(3);
  cv::split(HLS_img, HLS_channels);

  cv::Mat hue_adjusted = HLS_channels[0] + adj_h.mul(mask);
  hue_adjusted.forEach<float>([](float& p, const int* pos) -> void {
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
  cv::cvtColor(img, img, cv::COLOR_HLS2BGR);

  cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);

  return {std::move(img)};
}

auto HLSOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["target_hls"] = std::array<float, 3>{_target_HLS[0], _target_HLS[1], _target_HLS[2]};
  inner["hls_adj"] =
      std::array<float, 3>{_HLS_adjustment[0], _HLS_adjustment[1], _HLS_adjustment[2]};
  inner["h_range"] = _hue_range;
  inner["l_range"] = _lightness_range;
  inner["s_range"] = _saturation_range;

  o[_script_name]  = inner;
  return o;
}

void HLSOp::SetParams(const nlohmann::json& params) {
  if (params.contains(_script_name)) {
    nlohmann::json inner = params[_script_name];
    if (inner.contains("target_hls")) {
      auto tgt_hls = inner["target_hls"].get<std::array<float, 3>>();
      _target_HLS  = {tgt_hls[0], tgt_hls[1], tgt_hls[2]};
    }
    if (inner.contains("hls_adj")) {
      auto hls_adj    = inner["hls_adj"].get<std::array<float, 3>>();
      _HLS_adjustment = {hls_adj[0], hls_adj[1], hls_adj[2]};
    }
    if (inner.contains("h_range")) {
      _hue_range = inner["h_range"].get<float>();
    }
    if (inner.contains("l_range")) {
      _lightness_range = inner["l_range"].get<float>();
    }
    if (inner.contains("s_range")) {
      _saturation_range = inner["s_range"].get<float>();
    }
  }
}
};  // namespace puerhlab