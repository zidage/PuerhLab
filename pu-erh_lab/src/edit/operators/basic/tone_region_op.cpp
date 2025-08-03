#include "edit/operators/basic/tone_region_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <memory>
#include <string>

#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {

ToneRegionOp::ToneRegionOp(ToneRegion region) : _offset(0.0f), _region(region) { ComputeScale(); }

ToneRegionOp::ToneRegionOp(float offset, ToneRegion region) : _offset(offset), _region(region) {
  ComputeScale();
}

ToneRegionOp::ToneRegionOp(const nlohmann::json& params) { SetParams(params); }

/**
 * @brief Convert a region enum to its literal name
 *
 * @param region
 * @return std::string
 */
auto ToneRegionOp::RegionToString(ToneRegion region) -> std::string {
  switch (region) {
    case ToneRegion::BLACK:
      return "black";
    case ToneRegion::WHITE:
      return "white";
    case ToneRegion::SHADOWS:
      return "shadows";
    case ToneRegion::HIGHLIGHTS:
      return "highlights";
      break;
  }
  return "undefined";
}

/**
 * @brief Convert a region literal name to its enum class
 *
 * @param region
 * @return std::string
 */
auto ToneRegionOp::StringToRegion(std::string& region_str) -> ToneRegion {
  if (region_str == "black") return ToneRegion::BLACK;
  if (region_str == "white") return ToneRegion::WHITE;
  if (region_str == "shadows") return ToneRegion::SHADOWS;
  if (region_str == "highlights") return ToneRegion::HIGHLIGHTS;
  // Fallback solution
  return ToneRegion::BLACK;
}

/**
 * @brief Helper function to smooth highlights and shadows clamp
 *
 * @param edge0
 * @param edge1
 * @param x
 * @return float
 */
float SmoothStep(float edge0, float edge1, float x) {
  float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

/**
 * @brief Determine whether a luminance value lies within the region of the adjustment
 *
 * @param luminance
 * @return float
 */
auto ToneRegionOp::ComputeWeight(float luminance) const -> float {
  switch (_region) {
    case ToneRegion::BLACK:
      return std::pow(1.0f - luminance, 5.0f);
    case ToneRegion::WHITE:
      return std::pow(luminance, 4.0f);
    case ToneRegion::SHADOWS:
      return 1.0f - SmoothStep(0.1f, 0.5f, luminance);
    case ToneRegion::HIGHLIGHTS:
      return SmoothStep(0.7f, 1.0f, luminance);
    default:
      return 0.0f;
  }
}

/**
 * @brief Compute the scale from the offset
 *
 */
void ToneRegionOp::ComputeScale() {
  switch (_region) {
    case ToneRegion::BLACK:
    case ToneRegion::WHITE:
      _scale = _offset / 100.0f;
      break;
    case ToneRegion::SHADOWS:
      _scale = _offset / 300.0f;
      break;
    case ToneRegion::HIGHLIGHTS:
      _scale = _offset / 100.0f;
      break;
  }
}

auto ToneRegionOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& img = input.GetCPUData();
  if (img.depth() != CV_32F) {
    throw std::runtime_error("Tone region operator: Unsupported image format");
  }

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    for (int c = 0; c < 3; ++c) {
      float lum      = pixel[c];
      float weight   = ComputeWeight(lum);
      float push     = _scale * weight;

      // Limiter
      float max_push = 1.0f - lum;
      float min_push = -lum;
      push           = std::clamp(push, min_push, max_push);

      pixel[c] += push;
    }
  });

  return {std::move(img)};
}

auto ToneRegionOp::GetParams() const -> nlohmann::json {
  return {{GetScriptName(), {{"region", RegionToString(_region)}, {"offset", _offset}}}};
}

void ToneRegionOp::SetParams(const nlohmann::json& params) {
  const auto& inner      = params.at(_script_name);
  auto        region_str = inner.at("region").get<std::string>();
  _region                = StringToRegion(region_str);

  _offset                = inner.at("offset").get<float>();
  ComputeScale();
}
};  // namespace puerhlab