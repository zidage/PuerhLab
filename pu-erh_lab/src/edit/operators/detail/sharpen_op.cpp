#include "edit/operators/detail/sharpen_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "image/image_buffer.hpp"
#include "json.hpp"

namespace puerhlab {
SharpenOp::SharpenOp(float offset, float radius, float threshold)
    : _offset(offset), _radius(radius), _threshold(threshold) {
  ComputeScale();
}

void SharpenOp::ComputeScale() { _scale = _offset / 100.0f; }

auto SharpenOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["offset"]    = _offset;
  inner["radius"]    = _radius;
  inner["threshold"] = _threshold;

  o[_script_name]    = inner;
  return o;
}

void SharpenOp::SetParams(const nlohmann::json& params) {
  nlohmann::json inner = params[_script_name].get<nlohmann::json>();
  if (inner.contains("offset")) {
    _offset = inner["offset"].get<float>();
  }
  if (inner.contains("radius")) {
    _radius = inner["radius"].get<float>();
  }
  if (inner.contains("threshold")) {
    _threshold = inner["threshold"].get<float>();
  }
  ComputeScale();
}

auto SharpenOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& img = input.GetCPUData();
  cv::Mat  blurred, highpass, scaled, sharpened;

  cv::GaussianBlur(img, blurred, cv::Size(), _radius);
  cv::subtract(img, blurred, highpass);

  highpass.convertTo(scaled, highpass.type(), _scale);

  cv::add(img, scaled, sharpened);

  cv::Mat dst;
  if (_threshold > 0.0f) {
    cv::Mat diff, mask;
    cv::absdiff(img, blurred, diff);

    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);

    cv::threshold(diff, mask, _threshold, 1.0f, cv::THRESH_BINARY);

    sharpened.copyTo(dst, mask);
    img.copyTo(dst, 1.0f - mask);
  } else {
    dst = sharpened;
  }
  return {std::move(dst)};
}
};  // namespace puerhlab