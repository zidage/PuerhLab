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
    : _target_HLS(0, 0.5f, 1.0f),
      _HLS_adjustment(0.0f, 0.0f, 0.0f),
      _hue_range(15.0f),
      _lightness_range(0.1f),
      _saturation_range(0.1f) {}

HLSOp::HLSOp(const nlohmann::json& params) { SetParams(params); }

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

void HLSOp::Apply(std::shared_ptr<ImageBuffer> input) {
  if (cv::norm(_HLS_adjustment, cv::NORM_L2SQR) < 1e-10) {
    return;
  }

  cv::Mat& img = input->GetCPUData();
  cv::Mat  HLS_img;
  cv::cvtColor(img, HLS_img, cv::COLOR_RGB2HLS);

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

auto HLSOp::ToKernel() const -> Kernel {
  return Kernel{
      ._type = Kernel::Type::Point,
      ._func = [target_hls = _target_HLS, hls_adj = _HLS_adjustment, h_range = _hue_range,
                l_range = _lightness_range, s_range = _saturation_range](Pixel& in) {
        // Convert RGB to HLS
        // cv::Vec3f rgb(in.r, in.g, in.b);
        // cv::Mat   bgr_mat(1, 1, CV_32FC3);
        // cv::cvtColor(bgr_mat, bgr_mat, cv::COLOR_RGB2HLS);
        // cv::Vec3f hls      = bgr_mat.at<cv::Vec3f>(0, 0);

        float r = in.r, g = in.g, b = in.b;
        float max_c = std::max({r, g, b});
        float min_c = std::min({r, g, b});
        float L     = (max_c + min_c) * 0.5f;
        float H = 0.0f, S = 0.0f;
        float d = max_c - min_c == 0.0f ? 1e-10f : max_c - min_c;

        S       = (L < 0.5f) ? (d / (max_c + min_c)) : (d / (2.0f - max_c - min_c));
        if (max_c == r) {
          H = (g - b) / d + (g < b ? 6.0f : 0.0f);
        } else if (max_c == g) {
          H = (b - r) / d + 2.0f;
        } else if (max_c == b) {
          H = (r - g) / d + 4.0f;
        }
        H *= 60.0f;

        float target_h = target_hls[0];
        float target_l = target_hls[1];
        float target_s = target_hls[2];

        // Compute mask
        float h        = H;
        float l        = L;
        float s        = S;
        float hue_diff = std::abs(h - target_h);
        float hue_dist = std::min(hue_diff, 360.0f - hue_diff);

        float weight   = std::max(0.0f, 1.0f - hue_dist / h_range) *              // hue_w
                       std::max(0.0f, 1.0f - std::abs(l - target_l) / l_range) *  // lightness_w
                       std::max(0.0f, 1.0f - std::abs(s - target_s) / s_range);   // saturation_w

        float adj_h      = hls_adj[0];
        float adj_l      = hls_adj[1];
        float adj_s      = hls_adj[2];

        float h_adjusted = std::fmod(h + adj_h * weight, 360.0f);
        if (h_adjusted < 0) h_adjusted += 360.0f;

        float l_adjusted = std::clamp(l + adj_l * weight, 0.0f, 1.0f);
        float s_adjusted = std::clamp(s + adj_s * weight, 0.0f, 1.0f);

        // Convert HLS back to RGB
        if (s_adjusted == 0.0f) {
          return Pixel{l_adjusted, l_adjusted, l_adjusted};
        } else {
          float q       = (l_adjusted < 0.5f) ? (l_adjusted * (1.0f + s_adjusted))
                                              : (l_adjusted + s_adjusted - l_adjusted * s_adjusted);
          float p       = 2.0f * l_adjusted - q;

          auto  hue2rgb = [](float p, float q, float t) -> float {
            if (t < 0.0f) t += 1.0f;
            if (t > 1.0f) t -= 1.0f;
            if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
            if (t < 1.0f / 2.0f) return q;
            if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
            return p;
          };

          float r = hue2rgb(p, q, h_adjusted / 360.0f + 1.0f / 3.0f);
          float g = hue2rgb(p, q, h_adjusted / 360.0f);
          float b = hue2rgb(p, q, h_adjusted / 360.0f - 1.0f / 3.0f);
          in.r    = r;
          in.g    = g;
          in.b    = b;
        }
      }};
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