#pragma once

#include <cuda_runtime.h>
#include <device_types.h>
#include <texture_fetch_functions.h>
#include <vector_functions.h>

#include "edit/operators/op_kernel.hpp"
#include "param.cuh"

namespace puerhlab {
namespace CUDA {
// The following kernels always use 3D LUTs for color transformations.
struct GPU_TOWS_Kernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    float  u      = p->x * params.lut_max_coord_ws;
    float  v      = p->y * params.lut_max_coord_ws;
    float  w      = p->z * params.lut_max_coord_ws;

    float4 result = tex3D<float4>(params.to_ws_lut.texture_object, u, v, w);
    *p            = make_float4(result.x, result.y, result.z, p->w);
  }
};

struct GPU_LMT_Kernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    if (!params.lmt_enabled) return;

    float  u      = p->x * params.lut_max_coord_ws;
    float  v      = p->y * params.lut_max_coord_ws;
    float  w      = p->z * params.lut_max_coord_ws;

    float4 result = tex3D<float4>(params.lmt_lut.texture_object, u, v, w);
    *p            = make_float4(result.x, result.y, result.z, p->w);
  }
};

struct GPU_OUTPUT_Kernel : GPUPointOpTag {
  __device__ __forceinline__ void operator()(float4* p, GPUOperatorParams& params) const {
    float  u      = p->x * params.lut_max_coord_ws;
    float  v      = p->y * params.lut_max_coord_ws;
    float  w      = p->z * params.lut_max_coord_ws;

    float4 result = tex3D<float4>(params.to_output_lut.texture_object, u, v, w);
    *p            = make_float4(result.x, result.y, result.z, p->w);
  }
};
};  // namespace CUDA
};  // namespace puerhlab