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

auto ExposureOp::ToKernel() const -> Kernel {
  Kernel vec_kernel = {
      ._type     = Kernel::Type::Point,
      ._func     = PointKernelFunc([&s = _scale, &offset = _voffset](Pixel& in) {
        using namespace simple_simd;
        // PixelVec in_vec = PixelVec::Load(&in.r);
        // in_vec          = in_vec + offset;
        // in_vec.Store(&in.r);
        // __m128 voffset = _mm_set1_ps(s);
        // __m128 pin    = _mm_loadu_ps(&in.r);
        // pin           = _mm_add_ps(pin, offset);
        // _mm_storeu_ps(&in.r, pin);
        // f32x4 v = load_f32x4(&in.r);
        // v = add_f32x4(v, offset);
        // store_f32x4(&in.r, v);
        in.r += s;
        in.g += s;
        in.b += s;
      }),
      ._vec_func = VectorKernelFunc([&offset = _vec_offset](PixelVec& in) { in = in + offset; }),
  };

  return vec_kernel;
}

auto ExposureOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _exposure_offset;

  return o;
}

void ExposureOp::SetParams(const nlohmann::json& params) {
  _exposure_offset = params[GetScriptName()];
  _scale           = _exposure_offset / 17.52f;
  _vec_offset      = PixelVec(_scale);
  _voffset         = simple_simd::set1_f32(_scale);
}

};  // namespace puerhlab