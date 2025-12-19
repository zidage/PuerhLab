#pragma once

#include <hwy/highway.h>

#include <variant>

#include "edit/operators/op_kernel.hpp"
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
  // uint8_t*    _ptr;      // pointer to the top-left pixel of the tile
  // cv::Mat     tile_mat;  // Keep a reference to the tile data to manage its lifetime
  Pixel* _ptr;  // pointer to the top-left pixel of the Pixel tile
  int    _width, _height;
  int    _channels;  // number of channels
  int    _x0, _y0;   // position in the original image, (x0, y0) is the top-left corner
  int    _halo;      // number of pixels of halo (margin) around the tile
  int    original_width, original_height;  // original image size

  explicit Tile(int x0, int y0, int width, int height)
      : _width(width),
        _height(height),
        _channels(4),
        _x0(x0),
        _y0(y0),
        _halo(0),
        original_width(0),
        original_height(0) {
    _ptr = new Pixel[width * height];
  }

  ~Tile() { delete[] _ptr; }

  inline static auto CopyFrom(const cv::Mat& img_data, const Rect& region, int halo) -> Tile {
    auto     expanded_region = region.expand(halo);
    cv::Rect roi(expanded_region.x, expanded_region.y,
                 std::min(expanded_region.width, img_data.cols - expanded_region.x),
                 std::min(expanded_region.height, img_data.rows - expanded_region.y));
    cv::Mat  tile_mat = img_data(roi);
    Tile     tile(expanded_region.x, expanded_region.y, tile_mat.cols, tile_mat.rows);

    for (int i = 0; i < tile_mat.rows; ++i) {
      const float* src_row = tile_mat.ptr<const float>(i);
      for (int j = 0; j < tile_mat.cols; ++j) {
        Pixel& p = tile._ptr[i * tile_mat.cols + j];
        p.r      = src_row[j * tile_mat.channels() + 0];
        p.g      = src_row[j * tile_mat.channels() + 1];
        p.b      = src_row[j * tile_mat.channels() + 2];
        p.a      = 0.0f;
      }
    }

    return tile;
  }

  inline static auto CopyInto(cv::Mat& img_data, const Tile& tile) -> void {
    cv::Rect roi(tile._x0, tile._y0,
                 std::min(tile._width, img_data.cols - tile._x0),
                 std::min(tile._height, img_data.rows - tile._y0));
    cv::Mat  tile_mat = img_data(roi);

    for (int i = 0; i < tile_mat.rows; ++i) {
      float* dst_row = tile_mat.ptr<float>(i);
      for (int j = 0; j < tile_mat.cols; ++j) {
        const Pixel& p = tile.at(i, j);
        dst_row[j * tile_mat.channels() + 0] = p.r;
        dst_row[j * tile_mat.channels() + 1] = p.g;
        dst_row[j * tile_mat.channels() + 2] = p.b;
      }
    }
  }

  static auto CopyInto(ImageBuffer& img, const Tile& tile) -> void {
    auto& img_data = img.GetCPUData();
    CopyInto(img_data, tile);
  }

  static auto CopyFrom(ImageBuffer& img, const Rect& region, int halo) -> Tile {
    auto& img_data = img.GetCPUData();
    return CopyFrom(img_data, region, halo);
  }

  // Fast access without bounds checks; y-first to match row-major storage
  inline Pixel& at(int y, int x) { return _ptr[y * _width + x]; }
  inline const Pixel& at(int y, int x) const { return _ptr[y * _width + x]; }
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
};  // namespace puerhlab

HWY_BEFORE_NAMESPACE();
namespace puerhlab {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

class PixelVec {
 public:
  static constexpr size_t kLanes = 4;
  using D                        = HWY_CAPPED(float, kLanes);
  using V                        = hn::Vec<D>;

  HWY_INLINE PixelVec() : _v(Zero(D())) {}
  HWY_INLINE PixelVec(float scalar) : _v(Set(D(), scalar)) {}
  HWY_INLINE explicit PixelVec(V v) : _v(v) {}

  static HWY_INLINE PixelVec Load(const float* src) {
    // float tmp[4] = {src[0], src[1], src[2], 0.0f};
    return PixelVec(hn::Load(D(), src));
  }

  HWY_INLINE void Store(float* dst) const {
    // float tmp[4];
    // hn::Store(_v, D(), tmp);
    // dst[0] = tmp[0];
    // dst[1] = tmp[1];
    // dst[2] = tmp[2];
    hn::Store(_v, D(), dst);
  }

  HWY_INLINE PixelVec operator+(const PixelVec& other) const { return PixelVec(Add(_v, other._v)); }
  HWY_INLINE PixelVec operator-(const PixelVec& other) const { return PixelVec(Sub(_v, other._v)); }
  HWY_INLINE PixelVec operator*(const PixelVec& other) const { return PixelVec(Mul(_v, other._v)); }
  HWY_INLINE PixelVec operator/(const PixelVec& other) const { return PixelVec(Div(_v, other._v)); }

  HWY_INLINE PixelVec operator*(float scalar) const { return PixelVec(Mul(_v, Set(D(), scalar))); }

  HWY_INLINE PixelVec Clamp01() const { return PixelVec(Min(Max(_v, Zero(D())), Set(D(), 1.0f))); }

  HWY_INLINE V        raw() const { return _v; }

 private:
  V _v;
};
};  // namespace HWY_NAMESPACE
};  // namespace puerhlab
HWY_AFTER_NAMESPACE();

namespace puerhlab {
using PixelVec           = HWY_NAMESPACE::PixelVec;
using PointKernelFunc    = std::function<void(Pixel&)>;
using VectorKernelFunc   = std::function<void(PixelVec&)>;
using NeighborKernelFunc = std::function<ImageAccessor(ImageAccessor&)>;
using KernelFunc         = std::variant<PointKernelFunc>;

struct Kernel {
  enum class Type { Point, Neighbor } _type;

  PointKernelFunc  _func;
  VectorKernelFunc _vec_func;

  bool             has_vector_func;
  int              vector_length;  // length of the vector for vectorized operations
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
