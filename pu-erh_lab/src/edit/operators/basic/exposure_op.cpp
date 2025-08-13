#include "edit/operators/basic/exposure_op.hpp"

#include <opencv2/core/mat.hpp>
#include <utility>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 */
ExposureOp::ExposureOp() : _exposure_offset(0.0f) {
  _scale        = 1.0f;
  _scale_factor = hw::Set(hw::ScalableTag<float>(), 1.0f);
}

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 * @param exposure_offset
 */
ExposureOp::ExposureOp(float exposure_offset) : _exposure_offset(exposure_offset) {
  _scale        = std::pow(2.0f, _exposure_offset);
  _scale_factor = hw::Set(hw::ScalableTag<float>(), _scale);
}

ExposureOp::ExposureOp(const nlohmann::json& params) {
  SetParams(params);
  _scale_factor = hw::Set(hw::ScalableTag<float>(), _scale);
}

auto ExposureOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat&                     img              = input.GetCPUData();

  float*                       img_data         = reinterpret_cast<float*>(img.data);
  int                          total_floats_img = static_cast<int>(img.total() * img.channels());
  const hw::ScalableTag<float> d;
  int                          lanes = static_cast<int>(hw::Lanes(d));

  // For all tone regions, we can directly apply the adjustment
  // using a tone curve.
  cv::parallel_for_(
      cv::Range(0, total_floats_img),
      [&](const cv::Range& range) {
        int i           = range.start;
        int end         = range.end;

        int aligned_end = i + ((end - i) / lanes) * lanes;
        for (; i < aligned_end; i += lanes) {
          auto v_img = hw::Load(d, img_data + i);
          v_img      = hw::Mul(v_img, _scale_factor);
          v_img      = hw::Clamp(v_img, _min, _max);
          hw::Store(v_img, d, img_data + i);
        }

        for (; i < end; ++i) {
          img_data[i] = img_data[i] * _scale;
        }
      },
      cv::getNumThreads() * 4);
  return {std::move(img)};
}

auto ExposureOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _exposure_offset;

  return o;
}

void ExposureOp::SetParams(const nlohmann::json& params) {
  _exposure_offset = params[GetScriptName()];
  _scale           = std::pow(2.0f, _exposure_offset);
}

};  // namespace puerhlab