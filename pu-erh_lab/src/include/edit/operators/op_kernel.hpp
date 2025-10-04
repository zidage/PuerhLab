#pragma once

#include "image/image_buffer.hpp"

namespace puerhlab {
struct Pixel {
  float r, g, b;
};

using KernelFunc = std::function<Pixel(const Pixel&)>;

struct Kernel {
  KernelFunc _func;
};
};  // namespace puerhlab
