#include "edit/operators/wheel/color_wheel_op.hpp"

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
ColorWheelOpRegister::ColorWheelOpRegister() {
  OperatorFactory::Instance().Register(OperatorType::COLOR_WHEEL, [](const nlohmann::json& params) {
    return std::make_shared<ColorWheelOp>(params);
  });
}

static ColorWheelOpRegister _color_wheel_op_reg;

ColorWheelOp::ColorWheelOp()
    : _lift(), _gamma(), _gain(), _lift_crossover(0.25f), _gain_crossover(0.75f) {}

ColorWheelOp::ColorWheelOp(const nlohmann::json& params) { SetParams(params); }

void Clamp(cv::Mat& input) {
  cv::threshold(input, input, 1.0f, 1.0f, cv::THRESH_TRUNC);
  cv::threshold(input, input, 0.0f, 0.0f, cv::THRESH_TOZERO);
}

float bell(float L, float center, float width) {
  float x = (L - center) / width;
  return exp(-x * x);  // Gaussian
}

auto ColorWheelOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& img = input.GetCPUData();
  if (img.empty()) {
    throw std::invalid_argument("Color Wheel: Invalid input image");
  }

  // Get luminance graph
  cv::Mat img_Lab;
  cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> Lab_channels;
  cv::split(img_Lab, Lab_channels);
  cv::Mat   lightness = Lab_channels[0] / 100.0f;  // L

  // BGR
  cv::Vec3f lift_offset(_lift.color_offset.x + _lift.luminance_offset,
                        _lift.color_offset.y + _lift.luminance_offset,
                        _lift.color_offset.z + _lift.luminance_offset);
  cv::Vec3f gain_factor(_gain.color_offset.x + _gain.luminance_offset,
                        _gain.color_offset.y + _gain.luminance_offset,
                        _gain.color_offset.z + _gain.luminance_offset);
  cv::Vec3f gamma_inv(1.0f / (_gamma.color_offset.x + _gamma.luminance_offset),
                      1.0f / (_gamma.color_offset.y + _gamma.luminance_offset),
                      1.0f / (_gamma.color_offset.z + _gamma.luminance_offset));

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* pos) {
    float     L              = lightness.at<float>(pos[0], pos[1]);
    float     lift_w         = std::clamp(bell(L, 0.0f, 0.45f), 0.0f, 1.0f);
    float     gamma_w        = 1.0f;
    float     gain_w         = std::clamp(bell(L, 1.0f, 0.45f), 0.0f, 1.0f);

    // float total_w = lift_w + gamma_w + gain_w + 1e-6fff
    // lift_w /= total_w;
    // gamma_w = 1.0f;
    // gain_w /= total_w;

    cv::Vec3f original_pixel = pixel;
    cv::Vec3f lifted_pixel   = original_pixel + lift_offset;
    cv::Vec3f gained_pixel   = original_pixel.mul(gain_factor);
    cv::Vec3f gamma_pixel;
    gamma_pixel[0] = std::pow(original_pixel[0], gamma_inv[0]);
    gamma_pixel[1] = std::pow(original_pixel[1], gamma_inv[1]);
    gamma_pixel[2] = std::pow(original_pixel[2], gamma_inv[2]);
    pixel          = pixel + lift_w * (lifted_pixel - pixel) + gain_w * (gained_pixel - pixel) +
            gamma_w * (gamma_pixel - pixel);

    cv::saturate_cast<float>(pixel[0]);
    cv::saturate_cast<float>(pixel[1]);
    cv::saturate_cast<float>(pixel[2]);
  });

  return {std::move(img)};
}

auto ColorWheelOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json inner;

  inner["lift"]       = _lift;
  inner["gamma"]      = _gamma;
  inner["gain"]       = _gain;
  inner["crossovers"] = {{"lift", _lift_crossover}, {"gain", _gain_crossover}};

  o[_script_name]     = inner;
  return o;
}

void ColorWheelOp::SetParams(const nlohmann::json& params) {
  nlohmann::json inner = params.at(_script_name);
  if (inner.contains("lift")) inner.at("lift").get_to(_lift);
  if (inner.contains("gamma")) inner.at("gamma").get_to(_gamma);
  if (inner.contains("gain")) inner.at("gain").get_to(_gain);
  if (inner.contains("crossovers")) {
    const auto& crossovers = inner.at("crossovers");
    if (crossovers.contains("lift")) crossovers.at("lift").get_to(_lift_crossover);
    if (crossovers.contains("gain")) crossovers.at("gain").get_to(_gain_crossover);
  }
}
};  // namespace puerhlab