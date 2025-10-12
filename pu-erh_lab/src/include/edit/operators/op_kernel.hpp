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

struct Rect {
  int  x, y, width, height;  // (x, y) is the top-left corner

  Rect expand(int margin) const {
    Rect new_rect   = *this;
    new_rect.x      = std::max(0, x - margin);
    new_rect.y      = std::max(0, y - margin);
    new_rect.width  = width + 2 * margin;
    new_rect.height = height + 2 * margin;
    return new_rect;
  }
};

struct Tile {
  uint8_t*    _ptr;      // pointer to the top-left pixel of the tile
  cv::Mat     tile_mat;  // Keep a reference to the tile data to manage its lifetime
  int         _width, _height;
  int         _stride;    // byte per row
  int         _channels;  // number of channels
  int         _x0, _y0;   // position in the original image, (x0, y0) is the top-left corner
  int         _halo;      // number of pixels of halo (margin) around the tile
  int         original_width, original_height;  // original image size

  static auto CopyFrom(ImageBuffer& img, const Rect& region, int halo) -> Tile {
    Tile     tile;
    auto&    img_data        = img.GetCPUData();
    auto     expanded_region = region.expand(halo);
    cv::Rect roi(expanded_region.x, expanded_region.y,
                 std::min(expanded_region.width, img_data.cols - expanded_region.x),
                 std::min(expanded_region.height, img_data.rows - expanded_region.y));
    tile.tile_mat        = img_data(roi).clone();  // Clone to ensure data continuity
    tile._ptr            = tile.tile_mat.data;
    tile._width          = tile.tile_mat.cols;
    tile._height         = tile.tile_mat.rows;
    tile._stride         = static_cast<int>(tile.tile_mat.step);
    tile._channels       = tile.tile_mat.channels();
    tile._x0             = expanded_region.x;
    tile._y0             = expanded_region.y;
    tile._halo           = halo;
    tile.original_width  = img_data.cols;
    tile.original_height = img_data.rows;
    return tile;
  }

  static auto ViewFrom(ImageBuffer& img, Rect& region, int halo) -> Tile {
    return ViewFrom(img.GetCPUData(), region, halo);
  }

  static auto ViewFrom(const cv::Mat& img_data, Rect& region, int halo) -> Tile {
    Tile     tile;
    auto     expanded_region = region.expand(halo);
    cv::Rect roi(expanded_region.x, expanded_region.y,
                 std::min(expanded_region.width, img_data.cols - expanded_region.x),
                 std::min(expanded_region.height, img_data.rows - expanded_region.y));
    tile.tile_mat        = img_data(roi);  // View, no clone
    tile._ptr            = tile.tile_mat.data;
    tile._width          = tile.tile_mat.cols;
    tile._height         = tile.tile_mat.rows;
    tile._stride         = static_cast<int>(tile.tile_mat.step);
    tile._channels       = tile.tile_mat.channels();
    tile._x0             = expanded_region.x;
    tile._y0             = expanded_region.y;
    tile._halo           = halo;
    tile.original_width  = img_data.cols;
    tile.original_height = img_data.rows;
    return tile;
  }
};

struct ImageAccessor {
  Tile* _tile;

  ImageAccessor(Tile* tile) : _tile(tile) {}
  /**
   * @brief Access pixel at (x, y) with border replication
   *
   * @param x
   * @param y
   * @return Pixel
   */
  Pixel at(int x, int y) const {
    // Use border replication for out-of-bounds access
    x          = std::clamp(x, 0, _tile->original_width - 1);
    y          = std::clamp(y, 0, _tile->original_height - 1);
    float* row = reinterpret_cast<float*>(_tile->_ptr + (y - _tile->_y0) * _tile->_stride);
    float* p   = row + (x - _tile->_x0) * _tile->_channels;
    return Pixel{p[0], p[1], p[2]};
  }
};

using PointKernelFunc    = std::function<void(Pixel&)>;
using VectorKernelFunc   = std::function<void(const float* src, float* dst, int length)>;
using NeighborKernelFunc = std::function<ImageAccessor(ImageAccessor&)>;
using KernelFunc         = std::variant<PointKernelFunc>;

struct Kernel {
  enum class Type { Point, Vector, Neighbor } _type;

  PointKernelFunc _func;
};

struct KernelStream {
  std::vector<Kernel> _kernels;

  bool                AddToStream(const Kernel& kernel) {
    // Ensure all kernels in the stream are of the same type
    if (kernel._type == Kernel::Type::Point) {
      _kernels.push_back(kernel);
      return true;
    }
    return false;  // Type mismatch
  }

  void Clear() { _kernels.clear(); }
};
};  // namespace puerhlab
