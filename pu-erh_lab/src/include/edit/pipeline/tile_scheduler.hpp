#pragma once

#include <memory>

#include "concurrency/thread_pool.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {

class TileScheduler {
 private:
  KernelStream&                _stream;
  bool                         _ops_set = false;
  std::shared_ptr<ImageBuffer> _input_img;

  ThreadPool                   _thread_pool;
  int                       _total_tiles;   // total number of tiles
  int                       _tile_per_row;  // number of tiles per row
  int                       _tile_per_col;  // number of tiles per column
  int                       _tile_size;     // dynamic tile size based on image size

 public:
  TileScheduler() = delete;
  TileScheduler(std::shared_ptr<ImageBuffer> input_img, KernelStream& stream,
                int num_threads = 20);
  void SetInputImage(std::shared_ptr<ImageBuffer> img);
  auto ApplyOps() -> std::shared_ptr<ImageBuffer>;
  auto HasOps() const -> bool { return _stream._kernels.size() > 0; }
};
}  // namespace puerhlab