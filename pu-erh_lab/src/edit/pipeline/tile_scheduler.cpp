#include "edit/pipeline/tile_scheduler.hpp"

#include <cstddef>
#include <ctime>
#include <memory>

#include "edit/operators/op_kernel.hpp"
#include "image/image_buffer.hpp"
#include "utils/profiler/profiler.hpp"

namespace puerhlab {

TileScheduler::TileScheduler(std::shared_ptr<ImageBuffer> input_img, KernelStream& stream,
                             int num_threads)
    : _input_img(input_img), _stream(stream), _thread_pool(num_threads) {
  auto& input_buffer = _input_img->GetCPUData();
  // Determine tile size based on image dimensions
  // _tile_size         = std::max(input_buffer.cols, input_buffer.rows) / num_threads;
  _tile_size         = 49;  // Fixed tile size for better cache locality
  _tile_per_col      = std::ceil(static_cast<float>(input_buffer.cols) / _tile_size);
  _tile_per_row      = std::ceil(static_cast<float>(input_buffer.rows) / _tile_size);
  _total_tiles       = _tile_per_col * _tile_per_row;
}

void TileScheduler::SetInputImage(std::shared_ptr<ImageBuffer> img) { _input_img = img; }

auto TileScheduler::ApplyOps() -> std::shared_ptr<ImageBuffer> {
  using clock = std::chrono::high_resolution_clock;
  auto start  = clock::now();

  if (!_input_img) {
    throw std::runtime_error("TileScheduler: Input image not set.");
  }
  if (_stream._kernels.empty()) {
    return _input_img;
  }

  // Use const& for the input buffer.
  const cv::Mat&          input_buffer = _input_img->GetCPUData();
  cv::Mat                 output_buffer{input_buffer.size(), input_buffer.type()};

  std::atomic<int>        tiles_completed = 0;
  std::mutex              mtx;
  std::condition_variable cv;
  const int               channels = input_buffer.channels();

  for (int tile_idx = 0; tile_idx < _total_tiles; ++tile_idx) {
    _thread_pool.Submit(
        [this, tile_idx, &input_buffer, &output_buffer, &tiles_completed, &mtx, &cv, channels]() {
          // Get tile's starting coordinates
          // EASY_BLOCK("Tile Processing");
          int          tile_x = (tile_idx % _tile_per_col) * _tile_size;
          int          tile_y = (tile_idx / _tile_per_col) * _tile_size;

          // Define the tile's region of interest (ROI), clamping to image boundaries
          int          height = std::min(_tile_size, input_buffer.rows - tile_y);
          int          width  = std::min(_tile_size, input_buffer.cols - tile_x);

          Tile         tile   = Tile::CopyFrom(input_buffer, {tile_x, tile_y, width, height}, 0);

          Kernel::Type last_kernel_type = Kernel::Type::Init;
          for (int it_idx = 0; it_idx < (int)_stream._kernels.size(); ++it_idx) {
            Kernel& kernel = _stream._kernels[it_idx];
            // Check for kernel type consistency
            // Basic logic: Point kernels can be grouped together, neighbor kernels are applied
            // separately
            if (last_kernel_type == kernel._type) {
              continue;
            } else if (kernel._type == Kernel::Type::Point) {
              for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                  // Work on the inner (non-halo) region
                  Pixel& out = tile.at_inner(i, j);
                  // Apply kernel stream to the tile (this logic is unchanged)
                  for (int kt_idx = it_idx; kt_idx < (int)_stream._kernels.size(); ++kt_idx) {
                    Kernel& k = _stream._kernels[kt_idx];
                    if (k._type != kernel._type) {
                      break;
                    }
                    k._func(out);
                  }
                }
              }
            } else if (kernel._type == Kernel::Type::Neighbor) {
              // Apply neighbor kernel function to the entire tile
              for (int kt_idx = it_idx; kt_idx < (int)_stream._kernels.size(); ++kt_idx) {
                Kernel& k = _stream._kernels[kt_idx];
                if (k._type != kernel._type) {
                  break;
                }
                k._neighbor_func(tile);
              }
            }
            last_kernel_type = kernel._type;
          }

          Tile::CopyInto(output_buffer, tile);

          // Atomically signal completion and notify the main thread if all tasks are done
          if (tiles_completed.fetch_add(1, std::memory_order_release) + 1 == _total_tiles) {
            std::lock_guard<std::mutex> lock(mtx);
            cv.notify_one();
          }
          // EASY_END_BLOCK;
        });
  }

  // Wait efficiently for all tiles to be processed
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&]() { return tiles_completed.load(std::memory_order_acquire) == _total_tiles; });

  auto                                      end      = clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("TileScheduler: Processed %zu tiles in %.2f ms\n", _total_tiles, duration.count());
  return std::make_shared<ImageBuffer>(std::move(output_buffer));
}

}  // namespace puerhlab