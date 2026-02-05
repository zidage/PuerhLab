//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <opencv2/core/cuda_types.hpp>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"


namespace puerhlab {
namespace CUDA {

__constant__ float d_black_level[4];
__constant__ float d_wb_multipliers[4];

__constant__ int   remap[4] = {0, 1, 3, 2};
__global__ void ToLinearRefKernel(cv::cuda::PtrStep<float> image, int width, int height,
                                             float white_level_scale) {
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
  const int color_idx = remap[(((row % 2) * 2) + (col % 2)) % 4];



  // --- Start Processing ---

  // 1. Load pixel value and convert to float for processing
  float     pixel_val = image(row, col);

  // 2. Black Level Subtraction
  pixel_val -= d_black_level[color_idx];

  // The multipliers are normalized to the green channel (index 1)
  float       mask   = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
  const float wb_mul = (d_wb_multipliers[color_idx] / d_wb_multipliers[1]) * mask + (1.0f - mask);
  pixel_val *= wb_mul;

  // 4. White Level Scaling (Normalization)

  pixel_val /= white_level_scale;

  // 5. Clamp the result to the valid 16-bit range [0, 1.0f]
  pixel_val  = fmaxf(0.0f, pixel_val);

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

  if (raw_data.color.cblack[4] == 2 && raw_data.color.cblack[5] == 2) {
    for (unsigned int x = 0; x < raw_data.color.cblack[4]; ++x) {
      for (unsigned int y = 0; y < raw_data.color.cblack[5]; ++y) {
        const auto index   = y * 2 + x;
        black_level[index] = raw_data.color.cblack[6 + index];
      }
    }
  }

  return black_level;
}

static auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* {
  return raw_data.color.cam_mul;
}

void ToLinearRef(cv::cuda::GpuMat& image, LibRaw& raw_processor) {
  auto black_level = CalculateBlackLevel(raw_processor.imgdata.rawdata);
  auto wb          = GetWBCoeff(raw_processor.imgdata.rawdata);
  int w = image.cols;
  int h = image.rows;

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

    // 1. Copy black level and WB data from host to GPU's __constant__ memory.
    // This is very fast and efficient for small, read-only data.
    CUDA_CHECK(cudaMemcpyToSymbol(d_black_level, black_level.data(), sizeof(float) * 4));
    CUDA_CHECK(cudaMemcpyToSymbol(d_wb_multipliers, wb, sizeof(float) * 4));

    // 2. Define CUDA kernel launch grid dimensions
    const dim3 threads_per_block(32, 32);
    const dim3 num_blocks((image.cols + threads_per_block.x - 1) / threads_per_block.x,
                          (image.rows + threads_per_block.y - 1) / threads_per_block.y);

    // 3. Launch the kernel
    ToLinearRefKernel<<<num_blocks, threads_per_block>>>(image, image.cols, image.rows,
                                                                    maximum);

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