//  Copyright 2026 Yurun Zi
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

#include "decoders/processor/operators/gpu/cuda_rotate.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
namespace CUDA {
namespace {

template <typename T>
__global__ void Rotate90CWKernel(cv::cuda::PtrStepSz<T> src, cv::cuda::PtrStepSz<T> dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;  // dst col
  const int y = blockIdx.y * blockDim.y + threadIdx.y;  // dst row

  if (x >= dst.cols || y >= dst.rows) return;

  const int src_row = src.rows - 1 - x;
  const int src_col = y;
  dst(y, x)         = src(src_row, src_col);
}

template <typename T>
__global__ void Rotate90CCWKernel(cv::cuda::PtrStepSz<T> src, cv::cuda::PtrStepSz<T> dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;  // dst col
  const int y = blockIdx.y * blockDim.y + threadIdx.y;  // dst row

  if (x >= dst.cols || y >= dst.rows) return;

  const int src_row = x;
  const int src_col = src.cols - 1 - y;
  dst(y, x)         = src(src_row, src_col);
}

template <typename T>
static void Rotate90CW_Impl(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                            cv::cuda::Stream& stream) {
  dst.create(src.cols, src.rows, src.type());
  cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

  const dim3 block(32, 8);
  const dim3 grid((dst.cols + block.x - 1) / block.x, (dst.rows + block.y - 1) / block.y);
  Rotate90CWKernel<T><<<grid, block, 0, cuda_stream>>>(src, dst);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void Rotate90CCW_Impl(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                             cv::cuda::Stream& stream) {
  dst.create(src.cols, src.rows, src.type());
  cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

  const dim3 block(32, 8);
  const dim3 grid((dst.cols + block.x - 1) / block.x, (dst.rows + block.y - 1) / block.y);
  Rotate90CCWKernel<T><<<grid, block, 0, cuda_stream>>>(src, dst);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void Rotate90CW(cv::cuda::GpuMat& img) {
  if (img.empty()) return;

  cv::cuda::Stream stream;
  cv::cuda::GpuMat out;

  switch (img.type()) {
    case CV_32FC4:
      Rotate90CW_Impl<float4>(img, out, stream);
      break;
    case CV_32FC3:
      Rotate90CW_Impl<float3>(img, out, stream);
      break;
    default:
      CV_Error(cv::Error::StsUnsupportedFormat, "CUDA::Rotate90CW: unsupported type");
  }

  stream.waitForCompletion();
  img = out;
}

void Rotate90CCW(cv::cuda::GpuMat& img) {
  if (img.empty()) return;

  cv::cuda::Stream stream;
  cv::cuda::GpuMat out;

  switch (img.type()) {
    case CV_32FC4:
      Rotate90CCW_Impl<float4>(img, out, stream);
      break;
    case CV_32FC3:
      Rotate90CCW_Impl<float3>(img, out, stream);
      break;
    default:
      CV_Error(cv::Error::StsUnsupportedFormat, "CUDA::Rotate90CCW: unsupported type");
  }

  stream.waitForCompletion();
  img = out;
}

}  // namespace CUDA
}  // namespace puerhlab

