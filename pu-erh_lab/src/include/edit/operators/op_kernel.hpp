#pragma once

#include <variant>

#include "image/image_buffer.hpp"

namespace puerhlab {

struct Pixel {
  float r, g, b, a;

  Pixel operator*(float scalar) const {
    return Pixel{r * scalar, g * scalar, b * scalar, a * scalar};
  }
  Pixel operator+(const Pixel& other) const {
    return Pixel{r + other.r, g + other.g, b + other.b, a + other.a};
  }
  Pixel operator-(const Pixel& other) const {
    return Pixel{r - other.r, g - other.g, b - other.b, a - other.a};
  }
  Pixel& operator+=(const Pixel& other) {
    r += other.r;
    g += other.g;
    b += other.b;
    a += other.a;
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
  Pixel*                            _ptr   = nullptr;
  int                               _width = 0, _height = 0;
  int                               _channels = 4;

  int                               _x0 = 0, _y0 = 0;  // expanded tile origin in image coords
  int                               _halo     = 0;

  // Inner (non-halo) region in image coords + size
  int                               _inner_x0 = 0, _inner_y0 = 0;
  int                               _inner_width = 0, _inner_height = 0;

  // Inner (non-halo) region offset within the expanded tile (local coords)
  int                               _inner_off_x = 0, _inner_off_y = 0;

  static constexpr std::align_val_t kAlign{64};

  explicit Tile(int x0, int y0, int width, int height, int halo, int inner_x0, int inner_y0,
                int inner_width, int inner_height, int inner_off_x, int inner_off_y)
      : _width(width),
        _height(height),
        _x0(x0),
        _y0(y0),
        _halo(halo),
        _inner_x0(inner_x0),
        _inner_y0(inner_y0),
        _inner_width(inner_width),
        _inner_height(inner_height),
        _inner_off_x(inner_off_x),
        _inner_off_y(inner_off_y) {
    const size_t count = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    _ptr               = static_cast<Pixel*>(::operator new[](count * sizeof(Pixel), kAlign));
  }

  ~Tile() {
    if (_ptr) {
      ::operator delete[](_ptr, kAlign);
      _ptr = nullptr;
    }
  }

  Tile(const Tile&)            = delete;
  Tile& operator=(const Tile&) = delete;

  Tile(Tile&& other) noexcept { *this = std::move(other); }
  Tile& operator=(Tile&& other) noexcept {
    if (this == &other) return *this;
    // free current
    if (_ptr) ::operator delete[](_ptr, kAlign);

    // move fields
    _ptr          = other._ptr;
    _width        = other._width;
    _height       = other._height;
    _channels     = other._channels;
    _x0           = other._x0;
    _y0           = other._y0;
    _halo         = other._halo;
    _inner_x0     = other._inner_x0;
    _inner_y0     = other._inner_y0;
    _inner_width  = other._inner_width;
    _inner_height = other._inner_height;
    _inner_off_x  = other._inner_off_x;
    _inner_off_y  = other._inner_off_y;

    // null out source
    other._ptr    = nullptr;
    other._width = other._height = 0;
    other._inner_width = other._inner_height = 0;
    return *this;
  }

  inline static auto CopyFrom(const cv::Mat& img_data, const Rect& region, int halo) -> Tile {
    // Expect float image data (your code uses ptr<float>)
    CV_Assert(img_data.depth() == CV_32F);
    CV_Assert(img_data.channels() == 3 || img_data.channels() == 4);

    const Rect expanded = region.expand(halo);

    const int  roi_x    = expanded.x;
    const int  roi_y    = expanded.y;
    const int  roi_w    = std::min(expanded.width, img_data.cols - roi_x);
    const int  roi_h    = std::min(expanded.height, img_data.rows - roi_y);

    cv::Rect   roi(roi_x, roi_y, roi_w, roi_h);
    cv::Mat    tile_mat    = img_data(roi);

    // inner offset within expanded ROI (local coords)
    const int  inner_off_x = region.x - roi_x;
    const int  inner_off_y = region.y - roi_y;

    Tile tile(roi_x, roi_y, tile_mat.cols, tile_mat.rows, halo, region.x, region.y, region.width,
              region.height, inner_off_x, inner_off_y);

    const int ch = tile_mat.channels();
    for (int i = 0; i < tile_mat.rows; ++i) {
      const float* src_row = tile_mat.ptr<const float>(i);
      for (int j = 0; j < tile_mat.cols; ++j) {
        Pixel& p = tile._ptr[i * tile_mat.cols + j];
        p.r      = src_row[j * ch + 0];
        p.g      = src_row[j * ch + 1];
        p.b      = src_row[j * ch + 2];
        p.a      = (ch == 4) ? src_row[j * ch + 3] : 1.0f;
      }
    }
    return tile;  // safe (move-enabled, copy-disabled)
  }

  inline static auto CopyInto(cv::Mat& img_data, const Tile& tile) -> void {
    CV_Assert(img_data.depth() == CV_32F);
    CV_Assert(img_data.channels() == 3 || img_data.channels() == 4);

    // Write ONLY the inner region back to the image (simpler + avoids overlap confusion)
    const int dst_w = std::min(tile._inner_width, img_data.cols - tile._inner_x0);
    const int dst_h = std::min(tile._inner_height, img_data.rows - tile._inner_y0);
    if (dst_w <= 0 || dst_h <= 0) return;

    cv::Rect  inner_roi(tile._inner_x0, tile._inner_y0, dst_w, dst_h);
    cv::Mat   dst = img_data(inner_roi);

    const int ch  = dst.channels();
    for (int i = 0; i < dst_h; ++i) {
      float*    dst_row = dst.ptr<float>(i);
      const int src_y   = tile._inner_off_y + i;
      for (int j = 0; j < dst_w; ++j) {
        const int    src_x  = tile._inner_off_x + j;
        const Pixel& p      = tile.at(src_y, src_x);
        dst_row[j * ch + 0] = p.r;
        dst_row[j * ch + 1] = p.g;
        dst_row[j * ch + 2] = p.b;
        if (ch == 4) dst_row[j * ch + 3] = p.a;
      }
    }
  }

  inline Pixel&       at(int y, int x) { return _ptr[y * _width + x]; }
  inline const Pixel& at(int y, int x) const { return _ptr[y * _width + x]; }

  inline Pixel&       at_inner(int y, int x) {
    return _ptr[(y + _inner_off_y) * _width + (x + _inner_off_x)];
  }
  inline const Pixel& at_inner(int y, int x) const {
    return _ptr[(y + _inner_off_y) * _width + (x + _inner_off_x)];
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
    return _tile->_ptr[std::clamp(y, 0, _tile->_height - 1) * _tile->_width +
                       std::clamp(x, 0, _tile->_width - 1)];
  }
};

struct PointOpTag {};
struct NeighborOpTag {};

struct GPUPointOpTag {};
struct GPUNeighborOpTag {};


};  // namespace puerhlab
