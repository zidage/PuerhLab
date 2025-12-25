#pragma once

#include <cuda_runtime.h>
#include <device_types.h>

#include <cstddef>
#include <tuple>

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

template <typename... Ops>
struct GPU_PointChain {
  std::tuple<Ops...> _ops;

  GPU_PointChain(Ops... ops) : _ops(std::move(ops)...) {}

  template <size_t I = 0>
  __device__ __forceinline__ void ApplyOps(float4* p, GPUOperatorParams& params) {
    if constexpr (I < sizeof...(Ops)) {
      auto& op = std::get<I>(_ops);
      op(p, params);
      ApplyOps<I + 1>(p, params);
    }
  }
};

template <typename... Stages>
class GPU_StaticKernelStream {
  std::tuple<Stages...> _stages;

 public:
  GPU_StaticKernelStream(Stages... stages) : _stages(std::move(stages)...) {}

  template <size_t I = 0>
  inline float4* Dispatch(float4* current_src, float4* current_dst, int width, int height,
                          size_t pitch_elems, GPUOperatorParams& params, dim3 grid, dim3 block) {
    if constexpr (I < sizeof...(Stages)) {
      auto& stage      = std::get<I>(_stages);
      using StageType  = std::decay_t<decltype(stage)>;

      float4* next_src = nullptr;
      float4* next_dst = nullptr;

      if constexpr (HasApplyOps<StageType>::value) {
        // PointOp
        GenericPointKernel<<<grid, block>>>(stage, current_src, current_dst, width, height,
                                            pitch_elems, params);
        return Dispatch<I + 1>(current_src, current_dst, width, height, pitch_elems, params, grid,
                               block);
      } else {
        // NeighborOp
        GenericNeighborKernel<<<grid, block>>>(stage, current_src, current_dst, width, height,
                                               pitch_elems, params);
        next_src = current_dst;
        next_dst = current_src;
        return Dispatch<I + 1>(next_src, next_dst, width, height, pitch_elems, params, grid, block);
      }
    }

    return current_src;
  }

  void Process(float4* d_in, float4* d_temp, int width, int height, size_t pitch_elems,
               GPUOperatorParams& params) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    Dispatch<0>(d_in, d_temp, width, height, pitch_elems, params, grid, block);
  }
};
};  // namespace CUDA
};  // namespace puerhlab