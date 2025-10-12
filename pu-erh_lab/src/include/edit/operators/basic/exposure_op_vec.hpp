#pragma once

#include <hwy/base.h>

namespace puerhlab {
HWY_INLINE void CallPixelOffset(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                                size_t length, float offset);
}