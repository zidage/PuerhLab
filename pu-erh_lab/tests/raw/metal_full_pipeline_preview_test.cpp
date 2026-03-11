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
#include "edit/pipeline/pipeline_cpu.hpp"
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

}  // namespace

TEST(MetalFullPipelinePreview, DecodeGeometryAndMergedStageStillLife) {
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

  CPUPipelineExecutor pipeline;
  pipeline.SetForceCPUOutput(true);

  auto input  = std::make_shared<ImageBuffer>(std::move(raw_bytes));
  auto output = pipeline.Apply(input);

  ASSERT_NE(output, nullptr);
  if (!output->cpu_data_valid_) {
    ASSERT_NO_THROW(output->SyncToCPU());
  }

  const cv::Mat& full_cpu = output->GetCPUData();
  ASSERT_FALSE(full_cpu.empty());
  ASSERT_EQ(full_cpu.type(), CV_32FC4);
  ASSERT_EQ(full_cpu.channels(), 4);

  cv::Mat preview;
  cv::normalize(full_cpu, preview, 0.0, 1.0, cv::NORM_MINMAX);
  cv::cvtColor(preview, preview, cv::COLOR_RGBA2BGR);

  cv::namedWindow("Metal Full Pipeline Preview", cv::WINDOW_NORMAL);
  cv::imshow("Metal Full Pipeline Preview", preview);
  cv::waitKey(0);
#endif
}

}  // namespace puerhlab
