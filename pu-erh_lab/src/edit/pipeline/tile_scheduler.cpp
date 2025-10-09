#include "edit/pipeline/tile_scheduler.hpp"

#include <cstddef>
#include <memory>

#include "edit/operators/op_kernel.hpp"
#include "image/image_buffer.hpp"
#include "utils/queue/queue.hpp"

namespace puerhlab {

TileScheduler::TileScheduler(std::shared_ptr<ImageBuffer> input_img, KernelStream& stream,
                             size_t num_threads)
    : _input_img(input_img), _stream(stream), _thread_pool(num_threads) {
  auto& input_buffer = _input_img->GetCPUData();
  // Determine tile size based on image dimensions
  _tile_size         = std::max(input_buffer.cols, input_buffer.rows) / num_threads;
  _tile_per_col      = std::ceil(static_cast<float>(input_buffer.cols) / _tile_size);
  _tile_per_row      = std::ceil(static_cast<float>(input_buffer.rows) / _tile_size);
  _total_tiles       = _tile_per_col * _tile_per_row;
}

void TileScheduler::SetInputImage(std::shared_ptr<ImageBuffer> img) { _input_img = img; }

auto TileScheduler::ApplyOps() -> std::shared_ptr<ImageBuffer> {
  if (!_input_img) {
    throw std::runtime_error("TileScheduler: Input image not set.");
  }
  if (_stream._kernels.empty()) {
    return _input_img;
  }

  // Create output image buffer
  cv::Mat&                  input_buffer = _input_img->GetCPUData();
  cv::Mat                   output_buffer{input_buffer.size(), input_buffer.type()};

  // Split image into tiles and process in parallel
  BlockingMPMCQueue<size_t> tile_queue{_total_tiles};

  for (size_t tile_idx = 0; tile_idx < _total_tiles; ++tile_idx) {
    _thread_pool.Submit([this, tile_idx = tile_idx, input_buffer, output_buffer, &tile_queue]() {
      // Get each tile's x0 and y0
      size_t        tile_x = (tile_idx % _tile_per_col) * _tile_size;
      size_t        tile_y = (tile_idx / _tile_per_col) * _tile_size;
      if (tile_x >= input_buffer.cols || tile_y >= input_buffer.rows) {
        return;  // Out of bounds
      }

      Rect          tile_rect(tile_x, tile_y, std::min(_tile_size, input_buffer.cols - tile_x),
                              std::min(_tile_size, input_buffer.rows - tile_y));
      Tile          tile = Tile::ViewFrom(input_buffer, tile_rect, 0);
      ImageAccessor accessor(&tile);
      // Apply kernel stream to the tile
      for (int i = 0; i < tile._height; ++i) {
          float* row =
              reinterpret_cast<float*>(output_buffer.data + (tile._y0 + i) * output_buffer.step);
        for (int j = 0; j < tile._width; ++j) {
          Pixel out = accessor.at(tile._x0 +j, tile._y0 + i);
          for (Kernel& kernel : _stream._kernels) {
            auto func = std::get<PointKernelFunc>(kernel._func);
            out       = func(out);
          }
          // Write back to output buffer
          float* p = row + (tile._x0 + j) * tile._channels;
          p[0]     = out.r;
          p[1]     = out.g;
          p[2]     = out.b;
        }
      }
      tile_queue.push(tile_idx);
    });
  }

  size_t received_tiles = 0;
  while (true) {
    size_t recevied = tile_queue.pop();
    received_tiles++;
    if (received_tiles >= _total_tiles) {
      break;
    }
  }
  return std::make_shared<ImageBuffer>(std::move(output_buffer));
}

}  // namespace puerhlab