#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>

#include <cstddef>
#include <tuple>
#include <utility>

#include "edit/operators/GPU_kernels/param.cuh"

namespace puerhlab {
namespace CUDA {

template <typename T>
class HasApplyOps {
  template <typename U>
  static auto test(int)
      -> decltype(std::declval<U>().template ApplyOps<0>(nullptr,
                                                         std::declval<GPUOperatorParams&>()),
                  std::true_type());
  template <typename U>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename Op>
__global__ void GenericPointKernel(Op op, float4* __restrict src, float4* __restrict dst, int width,
                                   int height, size_t pitch_elems, GPUOperatorParams params) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    size_t  offset = y * pitch_elems + x;
    float4* ptr    = dst + offset;

    if (src != dst) {
      *ptr = src[offset];
    }

    // PointOp interface: operator()(float4*, params)
    op.template ApplyOps<0>(ptr, params);
  }
}

template <typename Op>
__global__ void GenericNeighborKernel(Op  op, const float4* __restrict src, float4* __restrict dst,
                                      int width, int height, size_t pitch_elems,
                                      GPUOperatorParams params) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // NeighborOp interface: operator()(x, y, src, dst, w, h, pitch, params)
    op(x, y, src, dst, width, height, pitch_elems, params);
  }
}

// Minimal tuple-like chain that is trivially device-callable; avoids std::get, which can be
// host-only in some libstdc++ builds used by nvcc.
template <typename... Ts>
struct OpList;

template <>
struct OpList<> {
  OpList() = default;
  __device__ __forceinline__ void Apply(float4*, GPUOperatorParams&) {}
};

template <typename Head, typename... Tail>
struct OpList<Head, Tail...> {
  Head                head;
  OpList<Tail...>     tail;

  OpList() = default;
  __host__            __device__ explicit OpList(Head h, Tail... t)
      : head(std::move(h)), tail(std::move(t)...) {}

  __device__ __forceinline__ void Apply(float4* p, GPUOperatorParams& params) {
    head(p, params);
    tail.Apply(p, params);
  }
};

template <typename... Ops>
struct GPU_PointChain {
  OpList<Ops...> _ops;

  GPU_PointChain(Ops... ops) : _ops(std::move(ops)...) {}

  template <size_t I = 0>
  __device__ __forceinline__ void ApplyOps(float4* p, GPUOperatorParams& params) {
    _ops.Apply(p, params);
  }
};

template <typename... Stages>
class GPU_StaticKernelStream {
  std::tuple<Stages...> _stages;

 public:
  GPU_StaticKernelStream(Stages... stages) : _stages(std::move(stages)...) {}

  template <size_t I = 0>
  inline void Dispatch(float4* src, float4* dst, int width, int height, size_t pitch_elems,
                       GPUOperatorParams& params, dim3 grid, dim3 block, cudaStream_t stream) {
    if constexpr (I < sizeof...(Stages)) {
      auto& stage     = std::get<I>(_stages);
      using StageType = std::decay_t<decltype(stage)>;

      if constexpr (HasApplyOps<StageType>::value) {
        // PointOp
        GenericPointKernel<<<grid, block, 0, stream>>>(stage, src, dst, width, height, pitch_elems,
                                                       params);
      } else {
        // NeighborOp
        GenericNeighborKernel<<<grid, block, 0, stream>>>(stage, src, dst, width, height,
                                                          pitch_elems, params);
      }

      std::swap(src, dst);  // Ping-pong buffers
      Dispatch<I + 1>(src, dst, width, height, pitch_elems, params, grid, block, stream);
    }
  }

  float4* Process(float4* d_in, float4* d_temp, int width, int height, size_t pitch_elems,
                  GPUOperatorParams& params, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    Dispatch<0>(d_in, d_temp, width, height, pitch_elems, params, grid, block, stream);
    cudaStreamSynchronize(stream);

    return (sizeof...(Stages) % 2 == 0) ? d_temp : d_in;
  }
};
};  // namespace CUDA
};  // namespace puerhlab