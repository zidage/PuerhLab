//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "edit/operators/operator_registeration.hpp"
#include "edit/pipeline/default_pipeline_params.hpp"
#include "edit/pipeline/pipeline_stage.hpp"
#include "image/image_buffer.hpp"
#include "type/type.hpp"

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

}  // namespace

TEST(MetalRawStagePreview, DecodeStillLifeWithRawStageOnly) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  const auto raw_path =
      std::filesystem::path(TEST_IMG_PATH) / "raw" / "still_life" / "DSC_2674.NEF";
  if (!std::filesystem::exists(raw_path)) {
    GTEST_SKIP() << "Sample RAW file is missing: " << raw_path.string();
  }

  auto raw_bytes = ReadFileToBuffer(raw_path);
  ASSERT_FALSE(raw_bytes.empty());

  RegisterAllOperators();

  OperatorParams global_params;
  PipelineStage  raw_stage(PipelineStageName::Image_Loading, false, false);

  nlohmann::json decode_params = pipeline_defaults::MakeDefaultRawDecodeParams();
  decode_params["raw"]["gpu_backend"]            = "gpu";
  decode_params["raw"]["backend"]                = "puerh";
  decode_params["raw"]["highlights_reconstruct"] = false;
  decode_params["raw"]["decode_res"]             = static_cast<int>(DecodeRes::FULL);
  raw_stage.SetOperator(OperatorType::RAW_DECODE, decode_params, global_params);

  auto input = std::make_shared<ImageBuffer>(std::move(raw_bytes));
  raw_stage.SetInputImage(input);

  auto output = raw_stage.ApplyStage(global_params);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(output->gpu_data_valid_);
  EXPECT_EQ(output->GetGPUType(), CV_32FC4);

  ASSERT_NO_THROW(output->SyncToCPU());
  const cv::Mat& raw_cpu = output->GetCPUData();
  ASSERT_FALSE(raw_cpu.empty());
  EXPECT_EQ(raw_cpu.type(), CV_32FC4);
  EXPECT_EQ(raw_cpu.channels(), 4);

  cv::Mat preview;
  cv::normalize(raw_cpu, preview, 0.0, 1.0, cv::NORM_MINMAX);
  cv::cvtColor(preview, preview, cv::COLOR_RGBA2BGR);

  cv::namedWindow("Metal RAW Stage Preview", cv::WINDOW_NORMAL);
  cv::imshow("Metal RAW Stage Preview", preview);
  cv::waitKey(0);
#endif
}

}  // namespace puerhlab
