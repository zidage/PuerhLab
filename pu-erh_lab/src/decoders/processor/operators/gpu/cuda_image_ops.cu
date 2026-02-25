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

#include "decoders/processor/operators/gpu/cuda_image_ops.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <stdexcept>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
namespace CUDA {
namespace {

__global__ void MergeRGBKernel(cv::cuda::PtrStepSz<float> red, cv::cuda::PtrStepSz<float> green,
                               cv::cuda::PtrStepSz<float> blue, cv::cuda::PtrStepSz<float3> out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out.cols || y >= out.rows) {
    return;
  }
  out(y, x) = make_float3(red(y, x), green(y, x), blue(y, x));
}

__global__ void RGBToRGBAKernel(cv::cuda::PtrStepSz<float3> in, cv::cuda::PtrStepSz<float4> out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out.cols || y >= out.rows) {
    return;
  }

  const float3 rgb = in(y, x);
  out(y, x) = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
}

auto GetCudaStream(cv::cuda::Stream* stream) -> cudaStream_t {
  if (stream == nullptr) {
    return nullptr;
  }
  return cv::cuda::StreamAccessor::getStream(*stream);
}

void MaybeSync(cudaStream_t stream) {
  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

}  // namespace

void MergeRGB(const cv::cuda::GpuMat& red, const cv::cuda::GpuMat& green,
              const cv::cuda::GpuMat& blue, cv::cuda::GpuMat& dst, cv::cuda::Stream* stream) {
  if (red.empty() || green.empty() || blue.empty()) {
    throw std::runtime_error("CUDA::MergeRGB: input channels must not be empty");
  }
  if (red.size() != green.size() || red.size() != blue.size()) {
    throw std::runtime_error("CUDA::MergeRGB: channel sizes must match");
  }
  if (red.type() != CV_32FC1 || green.type() != CV_32FC1 || blue.type() != CV_32FC1) {
    throw std::runtime_error("CUDA::MergeRGB: expected three CV_32FC1 channels");
  }

  dst.create(red.rows, red.cols, CV_32FC3);

  const dim3 block(32, 8);
  const dim3 grid((dst.cols + block.x - 1) / block.x, (dst.rows + block.y - 1) / block.y);

  const cudaStream_t cuda_stream = GetCudaStream(stream);
  MergeRGBKernel<<<grid, block, 0, cuda_stream>>>(red, green, blue, dst);
  CUDA_CHECK(cudaGetLastError());
  MaybeSync(cuda_stream);
}

void RGBToRGBA(cv::cuda::GpuMat& img, cv::cuda::Stream* stream) {
  if (img.empty()) {
    return;
  }
  if (img.type() == CV_32FC4) {
    return;
  }
  if (img.type() != CV_32FC3) {
    throw std::runtime_error("CUDA::RGBToRGBA: only CV_32FC3/CV_32FC4 are supported");
  }

  cv::cuda::GpuMat out(img.rows, img.cols, CV_32FC4);

  const dim3 block(32, 8);
  const dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);

  const cudaStream_t cuda_stream = GetCudaStream(stream);
  RGBToRGBAKernel<<<grid, block, 0, cuda_stream>>>(img, out);
  CUDA_CHECK(cudaGetLastError());
  MaybeSync(cuda_stream);

  img = out;
}

}  // namespace CUDA
}  // namespace puerhlab

