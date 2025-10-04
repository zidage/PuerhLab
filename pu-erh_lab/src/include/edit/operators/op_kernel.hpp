#pragma once

#include "image/image_buffer.hpp"

namespace puerhlab {
struct Pixel {
  float r, g, b;
};

struct Tile {
  uint8_t* ptr;  // pointer to the top-left pixel of the tile
  int      _width, _height;
  int      _stride;    // byte per row
  int      _channels;  // number of channels
  int      _x0, _y0;   // position in the original image
};

struct ImageAccessor {
  Tile* _tile;
  Pixel at(int x, int y) const {
    float* row = reinterpret_cast<float*>(_tile->ptr + (y - _tile->_y0) * _tile->_stride);
    float* p   = row + (x - _tile->_x0) * _tile->_channels;
    return Pixel{p[0], p[1], p[2]};
  }
};

using PointKernelFunc = std::function<Pixel(const Pixel&)>;
using NeighborKernelFunc = std::function<Pixel(const ImageAccessor&, int, int)>;
using KernelFunc = std::variant<PointKernelFunc, NeighborKernelFunc>;

struct Kernel {
  enum class Type { Point, Neighbor } _type;
  
  KernelFunc _func;
};
};  // namespace puerhlab
