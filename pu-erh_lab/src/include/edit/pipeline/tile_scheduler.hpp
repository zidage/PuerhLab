#pragma once

#include <memory>

#include "concurrency/thread_pool.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
template <typename KernelStreamT>
class StaticTileScheduler {
 private:
  std::shared_ptr<ImageBuffer> _input_img;
  KernelStreamT                _stream;

  ThreadPool                   _thread_pool;
  int                          _total_tiles;   // total number of tiles
  int                          _tile_per_row;  // number of tiles per row
  int                          _tile_per_col;  // number of tiles per column
  int                          _tile_size;     // dynamic tile size based on image size

 public:
  StaticTileScheduler() = delete;
  StaticTileScheduler(std::shared_ptr<ImageBuffer> input_img, KernelStreamT stream,
                      int num_threads = 20)
      : _input_img(input_img), _stream(std::move(stream)), _thread_pool(num_threads) {
    auto& input_buffer = _input_img->GetCPUData();
    // Determine tile size based on image dimensions
    // _tile_size         = std::max(input_buffer.cols, input_buffer.rows) / num_threads;
    _tile_size         = 64;  // Fixed tile size for better cache locality
    _tile_per_col      = static_cast<int>(std::ceil(static_cast<float>(input_buffer.cols) / _tile_size));
    _tile_per_row      = static_cast<int>(std::ceil(static_cast<float>(input_buffer.rows) / _tile_size));
    _total_tiles       = _tile_per_col * _tile_per_row;
  }
  void SetInputImage(std::shared_ptr<ImageBuffer> img) { _input_img = img; }
  auto ApplyOps(OperatorParams& params) -> std::shared_ptr<ImageBuffer> {
    // Similar implementation as TileScheduler but using static stream
    using clock = std::chrono::high_resolution_clock;
    auto start  = clock::now();

    if (!_input_img) {
      throw std::runtime_error("TileScheduler: Input image not set.");
    }

    // Use const& for the input buffer.
    const cv::Mat&          input_buffer = _input_img->GetCPUData();
    cv::Mat                 output_buffer{input_buffer.size(), input_buffer.type()};

    std::atomic<int>        tiles_completed = 0;
    std::mutex              mtx;
    std::condition_variable cv;
    const int               channels = input_buffer.channels();

    for (int tile_idx = 0; tile_idx < _total_tiles; ++tile_idx) {
      _thread_pool.Submit([this, &params = params, tile_idx, &input_buffer, &output_buffer,
                           &tiles_completed, &mtx, &cv, channels]() {
        // Get tile's starting coordinates
        // EASY_BLOCK("Tile Processing");
        int            tile_x = (tile_idx % _tile_per_col) * _tile_size;
        int            tile_y = (tile_idx / _tile_per_col) * _tile_size;

        // Define the tile's region of interest (ROI), clamping to image boundaries
        int            height = std::min(_tile_size, input_buffer.rows - tile_y);
        int            width  = std::min(_tile_size, input_buffer.cols - tile_x);

        Tile           tile   = Tile::CopyFrom(input_buffer, {tile_x, tile_y, width, height}, 0);

        OperatorParams params_copy = params;  // Copy params for thread safety
        _stream.ProcessTile(tile, params_copy);

        Tile::CopyInto(output_buffer, tile);

        // Atomically signal completion and notify the main thread if all tasks are done
        if (tiles_completed.fetch_add(1, std::memory_order_release) + 1 == _total_tiles) {
          std::lock_guard<std::mutex> lock(mtx);
          cv.notify_one();
        }
        // EASY_END_BLOCK;
      });
    }
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock,
            [&]() { return tiles_completed.load(std::memory_order_acquire) == _total_tiles; });

    auto                                      end      = clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("TileScheduler: Processed %d tiles in %.2f ms\n", _total_tiles, duration.count());
    return std::make_shared<ImageBuffer>(std::move(output_buffer));
  }
  auto HasOps() const -> bool { return true; }
};
};  // namespace puerhlab