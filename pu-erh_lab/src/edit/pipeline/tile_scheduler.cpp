#include "edit/pipeline/tile_scheduler.hpp"

#include <cstddef>
#include <ctime>
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
  // _tile_size         = std::max(input_buffer.cols, input_buffer.rows) / num_threads;
  _tile_size = 64;  // Fixed tile size for better cache locality
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

  std::atomic<size_t>     tiles_completed = 0;
  std::mutex              mtx;
  std::condition_variable cv;
  const int               channels = input_buffer.channels();

  for (size_t tile_idx = 0; tile_idx < _total_tiles; ++tile_idx) {
    _thread_pool.Submit(
        [this, tile_idx, &input_buffer, &output_buffer, &tiles_completed, &mtx, &cv, channels]() {
          // Get tile's starting coordinates
          size_t tile_x = (tile_idx % _tile_per_col) * _tile_size;
          size_t tile_y = (tile_idx / _tile_per_col) * _tile_size;

          // Define the tile's region of interest (ROI), clamping to image boundaries
          int height = std::min((int)_tile_size, input_buffer.rows - (int)tile_y);
          int width  = std::min((int)_tile_size, input_buffer.cols - (int)tile_x);

          for (int i = 0; i < height; ++i) {
            // Get raw pointers to the start of the current row
            const float* src_row = input_buffer.ptr<const float>(tile_y + i) + tile_x * channels;
            float*       dst_row = output_buffer.ptr<float>(tile_y + i) + tile_x * channels;

            for (int j = 0; j < width; ++j) {
              // Read input pixel directly
              // Pixel out{src_row[j * channels + 0], src_row[j * channels + 1],
                        // src_row[j * channels + 2]};
              // Add alpha channel if necessary: out.a = src_row[j * channels + 3];

              PixelVec in = PixelVec::Load(&src_row[j * channels + 0]);
              // Apply kernel stream to the tile (this logic is unchanged)
              for (Kernel& kernel : _stream._kernels) {
                // auto func = std::get<PointKernelFunc>(kernel._func);
                // TODO handle other kernel types
                kernel._vec_func(in);
              }

              // Write output pixel directly
              in.Store(&dst_row[j * channels + 0]);
              // dst_row[j * channels + 0] = out.r;
              // dst_row[j * channels + 1] = out.g;
              // dst_row[j * channels + 2] = out.b;
            }
          }

          // Atomically signal completion and notify the main thread if all tasks are done
          if (tiles_completed.fetch_add(1, std::memory_order_release) + 1 == _total_tiles) {
            std::lock_guard<std::mutex> lock(mtx);
            cv.notify_one();
          }
        });
  }

  // Wait efficiently for all tiles to be processed
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&]() { return tiles_completed.load(std::memory_order_acquire) == _total_tiles; });

  auto end = clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("TileScheduler: Processed %zu tiles in %.2f ms\n", _total_tiles, duration.count());
  return std::make_shared<ImageBuffer>(std::move(output_buffer));
}

}  // namespace puerhlab