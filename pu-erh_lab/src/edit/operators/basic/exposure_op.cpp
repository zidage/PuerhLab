#include "edit/operators/basic/exposure_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "edit/operators/op_kernel.hpp"
#include "image/image_buffer.hpp"
#include "utils/simd/simple_simd.hpp"

#if SIMPLE_SIMD_X86
#include <immintrin.h>
#include <xmmintrin.h>
#endif

namespace puerhlab {
// using hn = hwy::HWY_NAMESPACE;
/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 */
ExposureOp::ExposureOp() : _exposure_offset(0.0f) { _scale = 0.0f; }

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 * @param exposure_offset
 */
ExposureOp::ExposureOp(float exposure_offset) : _exposure_offset(exposure_offset) {
  _scale = _exposure_offset / 17.52f;
}

ExposureOp::ExposureOp(const nlohmann::json& params) { SetParams(params); }

void ExposureOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    pixel[0] += _scale;
    pixel[1] += _scale;
    pixel[2] += _scale;
  });
}


auto ExposureOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _exposure_offset;

  return o;
}

void ExposureOp::SetParams(const nlohmann::json& params) {
  _exposure_offset = params[GetScriptName()];
  _scale           = _exposure_offset / 17.52f;
}

void ExposureOp::SetGlobalParams(OperatorParams& params) const {
  // Should only be called once SetParams has been called
  params.exposure_offset = _scale;
}

};  // namespace puerhlab