//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <opencv2/core/cuda_types.hpp>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"


namespace puerhlab {
namespace CUDA {

struct WBParams {
  float black_level[4];
  float wb_multipliers[4];
  int   black_tile_width;
  int   black_tile_height;
  float pattern_black[36];
};

__global__ void ToLinearRefKernel(cv::cuda::PtrStep<float> image, int width, int height,
                                  float white_level_scale, WBParams wb_params,
                                  BayerPattern2x2 pattern) {
  // Calculate the global x and y coordinates of the pixel for this thread
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check to avoid processing out-of-bounds pixels
  if (col >= width || row >= height) {
    return;
  }

  // Determine the color channel (0, 1, 3, 2) for the current pixel based on its position.
  // This standard calculation assumes a 2x2 Bayer pattern (like RGGB, GRBG, etc.).
  // The 'bayer_pattern_offset' helps align to the specific pattern from LibRaw's `idata.filters`.
  // The LibRaw COLOR(row, col) macro can often be simplified to this.
  const int color_idx = pattern.raw_fc[BayerCellIndex(row, col)];



  // --- Start Processing ---

  // 1. Load pixel value and convert to float for processing
  float     pixel_val = image(row, col);

  // 2. Black Level Subtraction
  float pattern_black = 0.0f;
  if (wb_params.black_tile_width > 0 && wb_params.black_tile_height > 0) {
    const int tile_y = row % wb_params.black_tile_height;
    const int tile_x = col % wb_params.black_tile_width;
    pattern_black =
        wb_params.pattern_black[tile_y * wb_params.black_tile_width + tile_x];
  }
  pixel_val -= wb_params.black_level[color_idx] + pattern_black;

  // The multipliers are normalized to the green channel (index 1)
  float       mask   = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
  const float wb_mul = (wb_params.wb_multipliers[color_idx] / wb_params.wb_multipliers[1]) * mask + (1.0f - mask);
  pixel_val *= wb_mul;

  // 4. White Level Scaling (Normalization)

  pixel_val /= white_level_scale;

  // 5. Clamp the result to the valid 16-bit range [0, 1.0f]
  // pixel_val  = fmaxf(0.0f, pixel_val);

  // 6. Store the final result back to the GpuMat, rounding correctly
  image(row, col) = pixel_val;
}

static auto CalculateBlackLevel(const libraw_rawdata_t& raw_data) -> std::array<float, 4> {
  const auto           base_black_level = static_cast<float>(raw_data.color.black);
  std::array<float, 4> black_level      = {
      base_black_level + static_cast<float>(raw_data.color.cblack[0]),
      base_black_level + static_cast<float>(raw_data.color.cblack[1]),
      base_black_level + static_cast<float>(raw_data.color.cblack[2]),
      base_black_level + static_cast<float>(raw_data.color.cblack[3])};

  return black_level;
}

static auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* {
  return raw_data.color.cam_mul;
}

void ToLinearRef(cv::cuda::GpuMat& image, LibRaw& raw_processor, const BayerPattern2x2& pattern) {
  auto black_level = CalculateBlackLevel(raw_processor.imgdata.rawdata);
  auto wb          = GetWBCoeff(raw_processor.imgdata.rawdata);

  if (raw_processor.imgdata.color.as_shot_wb_applied != 1) {
    for (int c = 0; c < 4; ++c) {
      black_level[c] /= 65535.0f;
    }
    float min     = black_level[0];

    float maximum = raw_processor.imgdata.rawdata.color.maximum / 65535.0f - min;

    // Ensure the input GpuMat has the correct type
    CV_Assert(image.type() == CV_16UC1);
    CV_Assert(maximum > 0);

    image.convertTo(image, CV_32FC1, 1.0f / 65535.0f);

    // --- Full Processing Path ---

    // 1. Build per-invocation WB parameters (passed as kernel args, thread-safe).
    WBParams wb_params = {};
    for (int c = 0; c < 4; ++c) {
      wb_params.black_level[c]    = black_level[c];
      wb_params.wb_multipliers[c] = wb[c];
    }
    wb_params.black_tile_width = 0;
    wb_params.black_tile_height = 0;
    const int tile_width = raw_processor.imgdata.rawdata.color.cblack[4];
    const int tile_height = raw_processor.imgdata.rawdata.color.cblack[5];
    const int entries = tile_width * tile_height;
    if (entries > 0 && entries <= 36) {
      wb_params.black_tile_width = tile_width;
      wb_params.black_tile_height = tile_height;
      for (int i = 0; i < entries; ++i) {
        wb_params.pattern_black[i] =
            static_cast<float>(raw_processor.imgdata.rawdata.color.cblack[6 + i]) / 65535.0f;
      }
    }

    // 2. Define CUDA kernel launch grid dimensions
    const dim3 threads_per_block(32, 32);
    const dim3 num_blocks((image.cols + threads_per_block.x - 1) / threads_per_block.x,
                          (image.rows + threads_per_block.y - 1) / threads_per_block.y);

    // 3. Launch the kernel
    ToLinearRefKernel<<<num_blocks, threads_per_block>>>(image, image.cols, image.rows, maximum,
                                                         wb_params, pattern);

    // Check for any kernel launch errors (important for debugging)
    CUDA_CHECK(cudaGetLastError());
    // Optionally wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

  } else {
    // --- Simplified Path: Only White Level Scaling ---
    // For this simple operation, OpenCV's built-in function is highly optimized.
    // It performs the operation: image = image * scale.
    // The conversion is done in-place.
    image.convertTo(image, CV_32FC1);
  }
}
};  // namespace GPU
};  // namespace puerhlab
