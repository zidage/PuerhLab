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

#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
namespace CUDA {
namespace {
constexpr float kMinGain = 1e-6f;

__constant__ float M_const[9];

static inline float SafeDivide(const float numerator, const float denominator) {
  return numerator / std::max(denominator, kMinGain);
}

static inline cv::Matx33f BuildDiagonal(const float r, const float g, const float b) {
  return cv::Matx33f(r, 0.f, 0.f, 0.f, g, 0.f, 0.f, 0.f, b);
}

static inline cv::Matx33f NormalizeMultipliers(const float* mul) {
  const float g = std::max(mul[1], kMinGain);
  return BuildDiagonal(SafeDivide(mul[0], g), 1.f, SafeDivide(mul[2], g));
}

static inline bool HasValidCamXyz(const float cam_xyz[][3]) {
  float abs_sum = 0.f;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const float v = cam_xyz[r][c];
      if (!std::isfinite(v)) return false;
      abs_sum += std::abs(v);
    }
  }
  return abs_sum > 0.f;
}

static inline bool HasValidRgbCam(const float rgb_cam[][4]) {
  float abs_sum = 0.f;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const float v = rgb_cam[r][c];
      if (!std::isfinite(v)) return false;
      abs_sum += std::abs(v);
    }
  }
  return abs_sum > 0.f;
}

static inline cv::Matx33f BuildCamMatrix(const float cam_xyz[][3]) {
  return cv::Matx33f(cam_xyz[0][0], cam_xyz[0][1], cam_xyz[0][2], cam_xyz[1][0], cam_xyz[1][1],
                     cam_xyz[1][2], cam_xyz[2][0], cam_xyz[2][1], cam_xyz[2][2]);
}

static inline cv::Matx33f BuildCamToSrgbMatrix(const float rgb_cam[][4]) {
  return cv::Matx33f(rgb_cam[0][0], rgb_cam[0][1], rgb_cam[0][2], rgb_cam[1][0], rgb_cam[1][1],
                     rgb_cam[1][2], rgb_cam[2][0], rgb_cam[2][1], rgb_cam[2][2]);
}

static inline cv::Matx33f ComputeCam2Xyz(const cv::Matx33f& normalized_pre_mul,
                                         const cv::Matx33f& cam_xyz_matrix) {
  const cv::Matx33f xyz_to_cam = normalized_pre_mul * cam_xyz_matrix;
  return xyz_to_cam.inv();
}

__global__ void ApplyColorMatrixKernel(const uchar* srcptr, uchar* dstptr, int rows, int cols,
                                      size_t src_step, size_t dst_step) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) return;

  const float* src_p = (const float*)(srcptr + y * src_step) + x * 3;
  float*       dst_p = (float*)(dstptr + y * dst_step) + x * 3;

  const float  r     = src_p[0];
  const float  g     = src_p[1];
  const float  b     = src_p[2];

  dst_p[0]           = M_const[0] * r + M_const[1] * g + M_const[2] * b;
  dst_p[1]           = M_const[3] * r + M_const[4] * g + M_const[5] * b;
  dst_p[2]           = M_const[6] * r + M_const[7] * g + M_const[8] * b;
}

static void ApplyColorMatrix_Helper(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                                    const cv::Mat& matrix, cv::cuda::Stream& stream) {
  CV_Assert(src.type() == CV_32FC3);
  CV_Assert(matrix.isContinuous() && matrix.rows == 3 && matrix.cols == 3 && matrix.type() == CV_32F);

  if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
    dst.create(src.size(), src.type());
  }

  cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
  CUDA_CHECK(cudaMemcpyToSymbolAsync(M_const, matrix.data, 9 * sizeof(float), 0,
                                     cudaMemcpyHostToDevice, cuda_stream));

  const dim3 block(32, 32);
  const dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
  ApplyColorMatrixKernel<<<grid, block, 0, cuda_stream>>>(src.data, dst.data, src.rows, src.cols,
                                                         src.step, dst.step);
  CUDA_CHECK(cudaGetLastError());
}

static inline cv::Matx33f BuildTotalMatrix(const float rgb_cam[][4], const float* pre_mul,
                                          const float cam_xyz[][3]) {
  static const cv::Matx33f M_CAT_D65_to_D60 = {1.01303491f,  0.00610526f, -0.01497094f,
                                               0.00769823f,  0.99816335f, -0.00503204f,
                                               -0.00284132f, 0.00468516f, 0.92450614f};

  static const cv::Matx33f xyz2aces2065 = {1.0498110175f,  0.0000000000f, -0.0000974845f,
                                           -0.4959030231f, 1.3733130458f, 0.0982400361f,
                                           0.0000000000f,  0.0000000000f, 0.9912520182f};

  const cv::Matx33f pre_to_cam = BuildDiagonal(1.f, 1.f, 1.f);

  if (HasValidCamXyz(cam_xyz)) {
    const cv::Matx33f normalized_pre_mul = NormalizeMultipliers(pre_mul);
    const cv::Matx33f cam_xyz_matrix     = BuildCamMatrix(cam_xyz);
    const cv::Matx33f cam2xyz_matrix     = ComputeCam2Xyz(normalized_pre_mul, cam_xyz_matrix);
    return xyz2aces2065 * M_CAT_D65_to_D60 * cam2xyz_matrix * pre_to_cam;
  }

  if (!HasValidRgbCam(rgb_cam)) {
    throw std::runtime_error("CUDA::ApplyColorMatrix: missing both cam_xyz and rgb_cam matrices");
  }

  // Linear sRGB (D65) -> XYZ (D65)
  static const cv::Matx33f srgb2xyz_d65 = {0.41239080f, 0.35758434f, 0.18048079f, 0.21263901f,
                                           0.71516868f, 0.07219232f, 0.01933082f, 0.11919478f,
                                           0.95053215f};

  static const cv::Matx33f srgb2aces2065_d60 = xyz2aces2065 * M_CAT_D65_to_D60 * srgb2xyz_d65;
  const cv::Matx33f        cam2srgb_matrix   = BuildCamToSrgbMatrix(rgb_cam);
  return srgb2aces2065_d60 * cam2srgb_matrix * pre_to_cam;
}
}  // namespace

void ApplyColorMatrix(cv::cuda::GpuMat& img, const float rgb_cam[][4], const float* pre_mul,
                      const float* cam_mul, const float cam_xyz[][3]) {
  (void)cam_mul;  // Kept for signature parity with CPU backend.

  CV_Assert(img.type() == CV_32FC3);

  const cv::Matx33f total = BuildTotalMatrix(rgb_cam, pre_mul, cam_xyz);
  cv::cuda::Stream  stream;
  ApplyColorMatrix_Helper(img, img, cv::Mat(total), stream);
  stream.waitForCompletion();
}
};  // namespace CUDA
};  // namespace puerhlab
