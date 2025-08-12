#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {
  _scale = hw::Set(hw::ScalableTag<float>(), offset / 300.0f);
}

ShadowsOp::ShadowsOp(const nlohmann::json& params) {
  SetParams(params);
  _scale = hw::Set(hw::ScalableTag<float>(), _offset / 300.0f);
}

void ShadowsOp::GetMask(cv::Mat& src, cv::Mat& mask) {
  EASY_BLOCK("Get Shadow Mask");
  cv::Mat cached_mask;

  // Parameters
  float   percentile = 40.0f;  // shadows percentile
  float   transition = 20.0f;  // smooth zone width

  // Step 1: Downsample L channel before processing
  cv::Mat small_src;
  int     mask_scale = 32;  // 1/32 resolution
  cv::resize(src, small_src, cv::Size(src.cols / mask_scale, src.rows / mask_scale), 0, 0,
             cv::INTER_AREA);

  // Step 2: Histogram in [0, 100]
  int          histSize  = 8192;  // number of bins
  float        range[]   = {0.0f, 100.0f};
  const float* histRange = {range};
  cv::Mat      hist;
  cv::calcHist(&small_src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
  hist /= small_src.total();

  // Step 3: CDF
  std::vector<float> cdf(histSize);
  cdf[0] = hist.at<float>(0);
  for (int i = 1; i < histSize; ++i) cdf[i] = cdf[i - 1] + hist.at<float>(i);

  // Step 4: Threshold
  float threshold = 0.0f;
  for (int i = 0; i < histSize; ++i) {
    if (cdf[i] >= percentile / 100.0f) {
      threshold = range[0] + (range[1] - range[0]) * (i / static_cast<float>(histSize - 1));
      break;
    }
  }

  // Step 5: Low-res mask generation
  cv::Mat small_mask;
  if (transition <= 0.0f) {
    cv::threshold(small_src, small_mask, threshold, 1.0f, cv::THRESH_BINARY_INV);
  } else {
    cv::Mat temp;
    cv::subtract(small_src, threshold, temp);  // L - threshold
    temp /= transition;                        // normalize
    small_mask = 1.0f - temp;                  // invert

    small_mask *= 100.0f;

    cv::threshold(small_mask, small_mask, 100.0f, 100.0f, cv::THRESH_TRUNC);
    cv::threshold(small_mask, small_mask, 0.0f, 0.0f, cv::THRESH_TOZERO);
  }
  cv::GaussianBlur(small_mask, small_mask, cv::Size(7, 7), 0);

  // Step 6: Upsample back to original size
  cv::resize(small_mask, cached_mask, src.size(), 0, 0, cv::INTER_CUBIC);

  mask = cached_mask;
  // cv::Mat resized;
  // cv::resize(cached_mask, resized, cv::Size(512, 512));
  // cv::imshow("Shadow Mask", resized);
  // cv::waitKey(0);
  EASY_END_BLOCK;
}

auto ShadowsOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  auto scaled_luminance = hw::Div(luminance, _scale_factor);
  auto dist             = hw::Div(hw::Sub(scaled_luminance, _center), _knee);
  auto weight           = VExp_F32(hw::Mul(hw::Neg(dist), dist));

  auto delta            = hw::Mul(_scale, hw::Mul(weight, hw::Sub(_white, scaled_luminance)));
  auto output_luminance = hw::Mul(hw::Add(scaled_luminance, delta), _scale_factor);

  output_luminance      = hw::Clamp(output_luminance, _min, _max);

  return output_luminance;
}

auto ShadowsOp::GetOutput(float luminance) -> float {
  float       x      = luminance / 100.0f;

  const float center = 0.2f;

  float       dist   = (x - center) / 0.3f;  // knee = 0.15
  float       weight = std::exp(-dist * dist);
  float       delta  = _offset / 100.0f * weight * (1.0f - x);
  float       output = (x + delta) * 100.0f;

  return output;
}

auto ShadowsOp::GetScale() -> float { return _offset / 500.0f; }

auto ShadowsOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<ShadowsOp>::Apply(input);
}

auto ShadowsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void ShadowsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab