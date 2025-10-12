#include <hwy/base.h>
#if defined(EXPOSURE_OP_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef EXPOSURE_OP_INL_H_
#undef EXPOSURE_OP_INL_H_
#else
#define EXPOSURE_OP_INL_H_
#endif

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace puerhlab {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
HWY_MAYBE_UNUSED void ApplyExposureVec(const float* in, float* out, size_t process_length, float offset) {
  const hn::ScalableTag<float> d;        // tag for float vectors (scalable/portable)
  const auto voffset = Set(d, offset);   // vector filled with offset

  size_t i = 0;
  const size_t N = process_length;

  // Process full vector lanes
  const size_t lanes = Lanes(d);
  for (; i + lanes <= N; i += lanes) {
    const auto v = Load(d, in + i);      // aligned or unaligned load handled by Load
    const auto r = Add(v, voffset);      // r = v + offset
    Store(r, d, out + i);                // store back
  }

  // Tail: scalar fallback
  for (; i < N; ++i) {
    out[i] = in[i] + offset;
  }
}
}  // namespace HWY_NAMESPACE
}  // namespace puerhlab
HWY_AFTER_NAMESPACE();
#endif  // EXPOSURE_OP_INL_H_
  