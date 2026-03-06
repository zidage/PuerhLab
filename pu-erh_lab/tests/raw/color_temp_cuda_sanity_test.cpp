#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "edit/operators/GPU_kernels/param.cuh"
#include "edit/operators/basic/contrast_op.hpp"
#include "edit/operators/cst/odt_op.hpp"
#include "edit/operators/basic/color_temp_op.hpp"
#include "edit/operators/raw/raw_decode_op.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace {
auto ReadFileToBuffer(const std::filesystem::path& path) -> std::vector<uint8_t> {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }

  file.seekg(0, std::ios::end);
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size <= 0) {
    return {};
  }

  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return {};
  }
  return buffer;
}

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
}  // namespace

TEST(OutputTransformSanity, OpenDRTDefaultPresetResolvesFiniteRuntimeParams) {
  OutputTransformOp output_op(MakeOpenDRTParams());

  const auto params = output_op.GetParams().at("odt");
  EXPECT_EQ(params.at("method"), "open_drt");
  EXPECT_EQ(params.at("encoding_space"), "rec709");
  EXPECT_EQ(params.at("encoding_etof"), "gamma_2_2");
  EXPECT_EQ(params.at("limiting_space"), "rec709");
  ASSERT_TRUE(params.contains("open_drt"));
  EXPECT_EQ(params.at("open_drt").at("look_preset"), "standard");
  EXPECT_EQ(params.at("open_drt").at("display_encoding_preset"), "srgb_display");

  OperatorParams global_params;
  output_op.SetGlobalParams(global_params);

  EXPECT_EQ(global_params.to_output_params_.method_, ColorUtils::OutputTransformMethod::OPEN_DRT);
  EXPECT_EQ(global_params.to_output_params_.etof_, ColorUtils::ETOF::GAMMA_2_2);
  EXPECT_FLOAT_EQ(global_params.to_output_params_.display_linear_scale_, 1.0f);

  const auto& resolved = global_params.to_output_params_.open_drt_params_;
  EXPECT_TRUE(std::isfinite(resolved.peak_luminance_));
  EXPECT_TRUE(std::isfinite(resolved.ts_m2_));
  EXPECT_TRUE(std::isfinite(resolved.ts_s_));
  EXPECT_TRUE(std::isfinite(resolved.ts_s1_));
  EXPECT_TRUE(std::isfinite(resolved.creative_white_norm_));
  EXPECT_GT(resolved.peak_luminance_, 0.0f);
}

TEST(ContrastSanity, ZeroStrengthIsDisabledAndMinimumDoesNotBlackTheImage) {
  cv::Mat reference(1, 1, CV_32FC3, cv::Scalar(0.5f, 0.4f, 0.3f));

  ContrastOp     zero_contrast(nlohmann::json{{"contrast", 0.0f}});
  OperatorParams zero_params;
  zero_contrast.SetGlobalParams(zero_params);
  EXPECT_FALSE(zero_params.contrast_enabled_);
  EXPECT_FLOAT_EQ(zero_params.contrast_scale_, 0.0f);

  auto zero_image = std::make_shared<ImageBuffer>(reference.clone());
  zero_contrast.Apply(zero_image);
  const auto zero_pixel = zero_image->GetCPUData().at<cv::Vec3f>(0, 0);
  EXPECT_NEAR(zero_pixel[0], 0.5f, 1e-6f);
  EXPECT_NEAR(zero_pixel[1], 0.4f, 1e-6f);
  EXPECT_NEAR(zero_pixel[2], 0.3f, 1e-6f);

  ContrastOp     min_contrast(nlohmann::json{{"contrast", -100.0f}});
  OperatorParams min_params;
  min_contrast.SetGlobalParams(min_params);
  EXPECT_TRUE(min_params.contrast_enabled_);
  EXPECT_NE(min_params.contrast_scale_, 0.0f);

  auto min_image = std::make_shared<ImageBuffer>(reference.clone());
  min_contrast.Apply(min_image);
  const auto min_pixel = min_image->GetCPUData().at<cv::Vec3f>(0, 0);
  EXPECT_GT(cv::norm(min_pixel), 0.05f);
}

TEST(OutputTransformSanity, SwitchingMethodsClearsStaleGPUState) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  GPUOperatorParams gpu_params{};
  OperatorParams    cpu_params;

  OutputTransformOp aces_op(MakeACES2Params());
  aces_op.SetGlobalParams(cpu_params);
  gpu_params = GPUParamsConverter::ConvertFromCPU(cpu_params, gpu_params);
  EXPECT_EQ(gpu_params.to_output_params_.method_, GPU_OutputTransformMethod::ACES2);
  EXPECT_NE(gpu_params.to_output_params_.aces_odt_params_.table_reach_M_.texture_object_, 0);

  OutputTransformOp open_drt_op(MakeOpenDRTParams());
  open_drt_op.SetGlobalParams(cpu_params);
  gpu_params = GPUParamsConverter::ConvertFromCPU(cpu_params, gpu_params);
  EXPECT_EQ(gpu_params.to_output_params_.method_, GPU_OutputTransformMethod::OPEN_DRT);
  EXPECT_EQ(gpu_params.to_output_params_.aces_odt_params_.table_reach_M_.texture_object_, 0);
  EXPECT_TRUE(std::isfinite(gpu_params.to_output_params_.open_drt_params_.ts_m2_));

  aces_op.SetGlobalParams(cpu_params);
  gpu_params = GPUParamsConverter::ConvertFromCPU(cpu_params, gpu_params);
  EXPECT_EQ(gpu_params.to_output_params_.method_, GPU_OutputTransformMethod::ACES2);
  EXPECT_NE(gpu_params.to_output_params_.aces_odt_params_.table_reach_M_.texture_object_, 0);

  gpu_params.to_output_params_.aces_odt_params_.Reset();
  gpu_params.to_output_params_.open_drt_params_.Reset();
#endif
}

TEST(ColorTempCudaSanity, RawContextAndFallbackMatrixPath) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  const auto raw_path = std::filesystem::path(TEST_IMG_PATH) / "raw" / "_DSC0726.ARW";
  if (!std::filesystem::exists(raw_path)) {
    GTEST_SKIP() << "Sample RAW file is missing: " << raw_path.string();
  }

  std::vector<uint8_t> raw_bytes = ReadFileToBuffer(raw_path);
  ASSERT_FALSE(raw_bytes.empty());

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));

  nlohmann::json decode_params;
  decode_params["raw"] = {{"cuda", true},
                          {"highlights_reconstruct", false},
                          {"use_camera_wb", true},
                          {"backend", "puerh"},
                          {"decode_res", 1}};

  RawDecodeOp raw_decode_op(decode_params);
  EXPECT_NO_THROW(raw_decode_op.ApplyGPU(input));

  OperatorParams params;
  raw_decode_op.SetGlobalParams(params);
  EXPECT_TRUE(params.raw_runtime_valid_);
  EXPECT_EQ(params.raw_decode_input_space_, RawDecodeInputSpace::CAMERA);

  nlohmann::json color_temp_json;
  color_temp_json["color_temp"] = {{"mode", "as_shot"}, {"cct", 6500.0f}, {"tint", 0.0f}};
  ColorTempOp color_temp_op(color_temp_json);

  // SetGlobalParams now eagerly resolves matrices internally.
  color_temp_op.SetGlobalParams(params);
  EXPECT_TRUE(params.color_temp_matrices_valid_);
  EXPECT_TRUE(std::isfinite(params.color_temp_resolved_cct_));
  EXPECT_TRUE(std::isfinite(params.color_temp_resolved_tint_));

  params.raw_camera_make_         = "UnknownMake";
  params.raw_camera_model_        = "UnknownModel";
  params.color_temp_runtime_dirty_ = true;
  // Re-resolve via SetGlobalParams with unknown camera metadata.
  color_temp_op.SetGlobalParams(params);
  EXPECT_TRUE(params.color_temp_matrices_valid_);
#endif
}
}  // namespace puerhlab
