#include "edit/operators/color/tint_op.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {
TintOpRegister::TintOpRegister() {
  OperatorFactory::Instance().Register(OperatorType::TINT, [](const nlohmann::json& params) {
    return std::make_shared<TintOp>(params);
  });
}

static TintOpRegister tint_op_reg;

TintOp::TintOp() : _tint_offset(0.0f) { _scale = 0.0f; }

TintOp::TintOp(float tint_offset) : _tint_offset(tint_offset) {
  // In OpenCV, the value of a channel lies between -127 to 127
  _scale = tint_offset / 1000.0f;
}

TintOp::TintOp(const nlohmann::json& params) { SetParams(params); }

auto TintOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat&             img = input.GetCPUData();
  std::vector<cv::Mat> bgr_channels;

  cv::split(img, bgr_channels);

  bgr_channels[1] += _scale;
  // Thresholding
  cv::threshold(bgr_channels[1], bgr_channels[1], 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(bgr_channels[1], bgr_channels[1], 0.0f, 0.0f, cv::THRESH_TOZERO);

  cv::merge(bgr_channels, img);

  return {std::move(img)};
}

auto TintOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[_script_name] = _tint_offset;
  return o;
}

void TintOp::SetParams(const nlohmann::json& params) {
  if (params.contains(_script_name)) {
    _tint_offset = params.at(_script_name).get<float>();
    _scale       = _tint_offset / 1000.0f;
  } else {
    _tint_offset = 0.0f;
    _scale       = 0.0f;
  }
}

};  // namespace puerhlab