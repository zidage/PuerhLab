#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "edit/operators/GPU_kernels/cst.cuh"
#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/operators/cst/odt_op.hpp"

namespace puerhlab {
namespace {

auto MakeOpenDRTParams() -> nlohmann::json {
  return {{"odt",
           {{"method", "open_drt"},
            {"encoding_space", "rec709"},
            {"encoding_etof", "gamma_2_2"},
            {"limiting_space", "rec709"},
            {"peak_luminance", 100.0f},
            {"open_drt",
             {{"look_preset", "standard"},
              {"tonescale_preset", "use_look_preset"},
              {"display_encoding_preset", "srgb_display"},
              {"creative_white_preset", "use_look_preset"}}}}}};
}

auto MakeACES2Params() -> nlohmann::json {
  return {{"odt",
           {{"method", "aces2"},
            {"encoding_space", "rec709"},
            {"encoding_etof", "gamma_2_2"},
            {"limiting_space", "rec709"},
            {"peak_luminance", 100.0f}}}};
}

float HostAcesccEncode(float x) {
  constexpr float kLog2Min     = -15.0f;
  constexpr float kDenormTrans = 0.00003051757812f;
  constexpr float kDenormOff   = 0.00001525878906f;
  constexpr float kA           = 9.72f;
  constexpr float kB           = 17.52f;

  if (x <= 0.0f) {
    return (-16.0f + kA) / kB;
  }
  if (x < kDenormTrans) {
    return (std::log2(kDenormOff + x * 0.5f) + kA) / kB;
  }
  return (std::log2(x) + kA) / kB;
}

auto MakeEncodedSamples() -> std::array<float4, 3> {
  return {{
      make_float4(HostAcesccEncode(0.18f), HostAcesccEncode(0.18f), HostAcesccEncode(0.18f), 1.0f),
      make_float4(HostAcesccEncode(1.25f), HostAcesccEncode(0.20f), HostAcesccEncode(0.05f), 1.0f),
      make_float4(HostAcesccEncode(2.50f), HostAcesccEncode(0.80f), HostAcesccEncode(0.30f), 1.0f),
  }};
}

__global__ void RunOutputKernel(float4* pixels, int count, GPUOperatorParams params) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  CUDA::GPU_OUTPUT_Kernel kernel;
  kernel(&pixels[idx], params);
}

auto RunOutputTransform(const nlohmann::json& params_json) -> std::vector<float4> {
  OutputTransformOp output_op(params_json);
  OperatorParams    cpu_params;
  output_op.SetGlobalParams(cpu_params);

  GPUOperatorParams gpu_params{};
  gpu_params = GPUParamsConverter::ConvertFromCPU(cpu_params, gpu_params);

  const auto samples = MakeEncodedSamples();
  float4*    device_pixels = nullptr;
  cudaMalloc(&device_pixels, sizeof(float4) * samples.size());
  cudaMemcpy(device_pixels, samples.data(), sizeof(float4) * samples.size(), cudaMemcpyHostToDevice);

  RunOutputKernel<<<1, static_cast<unsigned int>(samples.size())>>>(
      device_pixels, static_cast<int>(samples.size()), gpu_params);
  cudaDeviceSynchronize();

  std::vector<float4> output(samples.size());
  cudaMemcpy(output.data(), device_pixels, sizeof(float4) * samples.size(), cudaMemcpyDeviceToHost);
  cudaFree(device_pixels);

  gpu_params.to_output_params_.aces_odt_params_.Reset();
  gpu_params.to_output_params_.open_drt_params_.Reset();
  return output;
}

void ExpectFinitePixels(const std::vector<float4>& pixels) {
  for (const auto& pixel : pixels) {
    EXPECT_TRUE(std::isfinite(pixel.x));
    EXPECT_TRUE(std::isfinite(pixel.y));
    EXPECT_TRUE(std::isfinite(pixel.z));
  }
}

}  // namespace

TEST(OutputTransformCudaSanity, BothMethodsProduceFiniteOutputForACESccInput) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  const auto open_drt_pixels = RunOutputTransform(MakeOpenDRTParams());
  const auto aces2_pixels    = RunOutputTransform(MakeACES2Params());

  ExpectFinitePixels(open_drt_pixels);
  ExpectFinitePixels(aces2_pixels);
#endif
}

}  // namespace puerhlab
