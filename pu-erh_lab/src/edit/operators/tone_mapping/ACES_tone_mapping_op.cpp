#include "edit/operators/tone_mapping/ACES_tone_mapping_op.hpp"

#include <opencv2/core.hpp>

namespace puerhlab {
ACESToneMappingOp::ACESToneMappingOp(const nlohmann::json& params) {}

void ACESToneMappingOp::CalculateOutput(cv::Vec3f& color, float adapted_lum) {
  const float A = 2.51f;
  const float B = 0.03f;
  const float C = 2.43f;
  const float D = 0.59f;
  const float E = 0.14f;

  color *= adapted_lum;
  // Apply the ACES tone mapping curve

  cv::Vec3f num   = color.mul(A * color + cv::Vec3f(B, B, B));
  cv::Vec3f denom = color.mul(C * color + cv::Vec3f(D, D, D)) + cv::Vec3f(E, E, E);
  cv::divide(num, denom, color);
};

auto ACESToneMappingOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& img = input.GetCPUData();

  img.forEach<cv::Vec3f>([this](cv::Vec3f& pixel, const int*) {
    // Calculate luminance
    float luminance   = 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
    // Adapted luminance
    float adapted_lum = std::max(luminance, 1e-6f);  // Avoid division by zero
    // Apply the ACES tone mapping curve
    CalculateOutput(pixel, adapted_lum);
  });

  return {std::move(img)};
}

void ACESToneMappingOp::SetParams(const nlohmann::json& params) {
  // Currently no parameters to set, but can be extended in the future
}
auto ACESToneMappingOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = {};  // No parameters for now
  return o;
}
};  // namespace puerhlab