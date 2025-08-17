#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "decoders/processor/cuda_operators.hpp"

__constant__ float M_const[9];

__global__ void    ApplyColorMatrixKernel(const uchar* srcptr, uchar* dstptr, int rows, int cols,
                                          size_t src_step, size_t dst_step) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) return;

  const float* src_p = (const float*)(srcptr + y * src_step) + x * 3;
  float*       dst_p = (float*)(dstptr + y * dst_step) + x * 3;

  float        r     = src_p[0];
  float        g     = src_p[1];
  float        b     = src_p[2];

  dst_p[0]           = M_const[0] * r + M_const[1] * g + M_const[2] * b;
  dst_p[1]           = M_const[3] * r + M_const[4] * g + M_const[5] * b;
  dst_p[2]           = M_const[6] * r + M_const[7] * g + M_const[8] * b;
}

void ApplyColorMatrix(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Mat& matrix,
                      cv::cuda::Stream& stream) {
  CV_Assert(src.type() == CV_32FC3);
  CV_Assert(matrix.isContinuous() && matrix.rows == 3 && matrix.cols == 3 &&
            matrix.type() == CV_32F);

  if (dst.empty() || dst.size() != src.size() || dst.type() != src.type())
    dst.create(src.size(), src.type());

  cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);

  cudaMemcpyToSymbolAsync(M_const, matrix.data, 9 * sizeof(float), 0, cudaMemcpyHostToDevice,
                          cudaStream);

  dim3 block(16, 16);
  dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

  ApplyColorMatrixKernel<<<grid, block, 0, cudaStream>>>(src.data, dst.data, src.rows, src.cols,
                                                         src.step, dst.step);
}

#define CUDA_CHECK(call)                                                                         \
  do {                                                                                           \
    cudaError_t err = call;                                                                      \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  } while (0)

// Use __constant__ memory for small arrays that are read by all threads.
// This is faster than global memory.
__constant__ float d_black_level[4];
__constant__ float d_wb_multipliers[4];
__constant__ int remap[4] = {0, 1, 3, 2}; 

/**
 * @brief CUDA kernel to perform black level subtraction, white balancing, and white level scaling.
 *
 * @param image The GpuMat data (pointer and step).
 * @param width Image width.
 * @param height Image height.
 * @param white_level_scale The scaling factor (65535.0f / maximum).
 * @param bayer_pattern_offset The offset to determine the starting color of the Bayer pattern.
 * (e.g., for RGGB, this would be the index of R).
 */
__global__ void WhiteBalanceCorrectionKernel(cv::cuda::PtrStep<ushort> image, int width, int height,
                                             float white_level_scale, int bayer_pattern_offset) {
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
  const int color_idx = remap[(((row % 2) * 2) + (col % 2) + bayer_pattern_offset) % 4];


  // Get a pointer to the current pixel
  ushort*   pixel_ptr = (ushort*)((char*)image.data + row * image.step) + col;

  // --- Start Processing ---

  // 1. Load pixel value and convert to float for processing
  float     pixel_val = static_cast<float>(*pixel_ptr);

  // 2. Black Level Subtraction
  pixel_val -= d_black_level[color_idx];

  // 3. White Balance Multiplication
  // The multipliers are normalized to the green channel (index 1)
  float mask = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
  const float wb_mul = (d_wb_multipliers[color_idx] / d_wb_multipliers[1]) * mask + (1.0f - mask);
  pixel_val *= wb_mul;

  // 4. White Level Scaling (Normalization)
  pixel_val *= white_level_scale;

  // 5. Clamp the result to the valid 16-bit range [0, 65535]
  pixel_val  = fmaxf(0.0f, fminf(65535.0f, pixel_val));

  // 6. Store the final result back to the GpuMat, rounding correctly
  *pixel_ptr = static_cast<ushort>(pixel_val);
}

/**
 * @brief C++ wrapper function to process a raw image on the GPU.
 *
 * @param image The cv::cuda::GpuMat to process (must be CV_16UC1).
 * @param black_level An array of 4 floats for black level correction.
 * @param wb_coeffs An array of 4 floats for white balance.
 * @param maximum The maximum possible pixel value from the sensor.
 * @param apply_wb_and_black_level A flag to control the processing path.
 * @param bayer_offset The starting filter color index (from libraw_data_t.idata.filters).
 */
void WhiteBalanceCorrection(cv::cuda::GpuMat& image, const std::array<float, 4>& black_level,
                            const float* wb_coeffs, float maximum, bool apply_wb_and_black_level,
                            int bayer_offset) {
  // Ensure the input GpuMat has the correct type
  CV_Assert(image.type() == CV_16UC1);
  CV_Assert(maximum > 0);

  const float white_level_scale = 65535.0f / maximum;

  if (apply_wb_and_black_level) {
    // --- Full Processing Path ---

    // 1. Copy black level and WB data from host to GPU's __constant__ memory.
    // This is very fast and efficient for small, read-only data.
    CUDA_CHECK(cudaMemcpyToSymbol(d_black_level, black_level.data(), sizeof(float) * 4));
    CUDA_CHECK(cudaMemcpyToSymbol(d_wb_multipliers, wb_coeffs, sizeof(float) * 4));

    // 2. Define CUDA kernel launch grid dimensions
    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks((image.cols + threads_per_block.x - 1) / threads_per_block.x,
                          (image.rows + threads_per_block.y - 1) / threads_per_block.y);

    // 3. Launch the kernel
    WhiteBalanceCorrectionKernel<<<num_blocks, threads_per_block>>>(
        image, image.cols, image.rows, white_level_scale, bayer_offset);

    // Check for any kernel launch errors (important for debugging)
    CUDA_CHECK(cudaGetLastError());
    // Optionally wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

  } else {
    // --- Simplified Path: Only White Level Scaling ---
    // For this simple operation, OpenCV's built-in function is highly optimized.
    // It performs the operation: image = image * scale.
    // The conversion is done in-place.
    image.convertTo(image, CV_16U, white_level_scale);
  }
}