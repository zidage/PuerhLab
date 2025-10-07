#pragma once

#include <variant>

#include "image/image_buffer.hpp"


namespace puerhlab {
struct Pixel {
  float r, g, b;

  Pixel operator*(float scalar) const { return Pixel{r * scalar, g * scalar, b * scalar}; }
  Pixel operator+(const Pixel& other) const { return Pixel{r + other.r, g + other.g, b + other.b}; }
  Pixel operator-(const Pixel& other) const { return Pixel{r - other.r, g - other.g, b - other.b}; }
  Pixel& operator+=(const Pixel& other) {
    r += other.r;
    g += other.g;
    b += other.b;
    return *this;
  }
};

struct Tile {
  uint8_t* ptr;  // pointer to the top-left pixel of the tile
  int      _width, _height;
  int      _stride;    // byte per row
  int      _channels;  // number of channels
  int      _x0, _y0;   // position in the original image, (x0, y0) is the top-left corner
  int      original_width, original_height;  // original image size
};

struct ImageAccessor {
  Tile* _tile;

  /**
   * @brief Access pixel at (x, y) with border replication
   * 
   * @param x 
   * @param y 
   * @return Pixel 
   */
  Pixel at(int x, int y) const {
    // Use border replication for out-of-bounds access
    x = std::clamp(x, 0, _tile->original_width - 1);
    y = std::clamp(y, 0, _tile->original_height - 1);
    float* row = reinterpret_cast<float*>(_tile->ptr + (y - _tile->_y0) * _tile->_stride);
    float* p   = row + (x - _tile->_x0) * _tile->_channels;
    return Pixel{p[0], p[1], p[2]};
  }
};

using PointKernelFunc    = std::function<Pixel(const Pixel&)>;
using NeighborKernelFunc = std::function<Pixel(const ImageAccessor&, int, int)>;
using KernelFunc         = std::variant<PointKernelFunc, NeighborKernelFunc>;

struct Kernel {
  enum class Type { Point, Neighbor } _type;

  KernelFunc _func;
};
};  // namespace puerhlab
