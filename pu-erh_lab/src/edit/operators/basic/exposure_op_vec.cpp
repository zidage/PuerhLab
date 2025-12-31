//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace puerhlab {
namespace HWY_NAMESPACE {
namespace {
namespace hn = hwy::HWY_NAMESPACE;
HWY_INLINE void PixelOffset(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, size_t length,
                            float offset) {
  const hn::ScalableTag<float> d;  // tag for float vectors (scalable/portable)
  const auto                   voffset = Set(d, offset);  // vector filled with offset

  size_t                       i       = 0;
  const size_t                 N       = length;

  // Process full vector lanes
  const size_t                 lanes   = Lanes(d);
  for (; i + lanes <= N; i += lanes) {
    const auto v = Load(d, in + i);  // aligned or unaligned load handled by Load
    const auto r = Add(v, voffset);  // r = v + offset
    Store(r, d, out + i);            // store back
  }

  // Tail: scalar fallback
  for (; i < N; ++i) {
    out[i] = in[i] + offset;
  }
}
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace puerhlab
HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE
namespace puerhlab {
HWY_INLINE void CallPixelOffset(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                                size_t length, float offset) {
  HWY_NAMESPACE::PixelOffset(in, out, length, offset);
}
}  // namespace puerhlab
#endif  // HWY_ONCE
