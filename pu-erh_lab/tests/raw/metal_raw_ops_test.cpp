//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/metal_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/metal_highlight_reconstruct.hpp"
#include "image/metal_image.hpp"
#include "metal/metal_utils/metal_convert_utils.hpp"

namespace puerhlab {
namespace {

auto MakeBayerPattern(int rows, int cols) -> cv::Mat {
  cv::Mat bayer(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    float* row = bayer.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      row[x] = static_cast<float>((y * 17 + x * 11) % 251) / 255.0f;
    }
  }
  return bayer;
}

auto MakeRGBAImage(int rows, int cols) -> cv::Mat {
  cv::Mat rgba(rows, cols, CV_32FC4);
  for (int y = 0; y < rows; ++y) {
    cv::Vec4f* row = rgba.ptr<cv::Vec4f>(y);
    for (int x = 0; x < cols; ++x) {
      row[x] = cv::Vec4f(static_cast<float>(x) / 17.0f, static_cast<float>(y) / 13.0f,
                         static_cast<float>(x + y) / 29.0f, 1.0f);
    }
  }
  return rgba;
}

auto CFAColorAt(int y, int x) -> int {
  return (y & 1) ? ((x & 1) ? 2 : 1) : ((x & 1) ? 1 : 0);
}

auto HighlightColorAt(int y, int x) -> int {
  return (y & 1) ? ((x & 1) ? 1 : 2) : ((x & 1) ? 1 : 0);
}

auto MakeClampedImage() -> cv::Mat {
  cv::Mat img(5, 6, CV_32FC1);
  for (int y = 0; y < img.rows; ++y) {
    float* row = img.ptr<float>(y);
    for (int x = 0; x < img.cols; ++x) {
      row[x] = -0.35f + static_cast<float>(y * img.cols + x) * 0.11f;
    }
  }
  return img;
}

auto MakeHighlightInput(int rows, int cols) -> cv::Mat {
  cv::Mat img(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    float* row = img.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      static constexpr float kBaseByColor[3] = {0.9810f, 0.9845f, 0.9790f};
      const int              color           = HighlightColorAt(y, x);
      float                  value           = kBaseByColor[color] + 0.0002f * float((x + y) % 5);
      if (x >= cols / 4 && x < (3 * cols) / 4 && y >= rows / 4 && y < (3 * rows) / 4) {
        value = 1.08f + 0.015f * float((x + y) % 3);
      }
      row[x] = value;
    }
  }
  return img;
}

void InitHighlightRawProcessor(LibRaw& raw_processor) {
  raw_processor.imgdata.color.cam_mul[0] = 2.15f;
  raw_processor.imgdata.color.cam_mul[1] = 1.0f;
  raw_processor.imgdata.color.cam_mul[2] = 1.42f;
  raw_processor.imgdata.color.cam_mul[3] = 1.0f;
}

}  // namespace

TEST(MetalRawOpsTest, CropRectMatchesCPUReference) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  const cv::Mat src = MakeRGBAImage(9, 11);
  const cv::Rect crop_rect(2, 3, 5, 4);

  metal::MetalImage image;
  image.Upload(src);

  metal::MetalImage cropped;
  ASSERT_NO_THROW(image.CropTo(cropped, crop_rect));

  cv::Mat cropped_cpu;
  cropped.Download(cropped_cpu);

  const cv::Mat expected = src(crop_rect).clone();
  ASSERT_EQ(cropped_cpu.type(), CV_32FC4);
  ASSERT_EQ(cropped_cpu.size(), expected.size());
  EXPECT_LE(cv::norm(cropped_cpu, expected, cv::NORM_INF), 1e-6);
#endif
}

TEST(MetalRawOpsTest, DebayerRcdProducesRGBAAndPreservesCFASamples) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  const cv::Mat bayer = MakeBayerPattern(18, 20);

  metal::MetalImage image;
  image.Upload(bayer);

  ASSERT_NO_THROW(metal::BayerRGGB2RGB_RCD(image));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  ASSERT_EQ(gpu_result.type(), CV_32FC4);
  ASSERT_EQ(gpu_result.size(), bayer.size());

  for (int y = 0; y < gpu_result.rows; ++y) {
    const float*      raw_row = bayer.ptr<float>(y);
    const cv::Vec4f* rgba_row = gpu_result.ptr<cv::Vec4f>(y);
    for (int x = 0; x < gpu_result.cols; ++x) {
      const cv::Vec4f px = rgba_row[x];
      const float     raw = raw_row[x];

      EXPECT_GE(px[0], 0.0f);
      EXPECT_GE(px[1], 0.0f);
      EXPECT_GE(px[2], 0.0f);
      EXPECT_NEAR(px[3], 1.0f, 1e-6);

      switch (CFAColorAt(y, x)) {
        case 0:
          EXPECT_NEAR(px[0], raw, 1e-6);
          break;
        case 1:
          EXPECT_NEAR(px[1], raw, 1e-6);
          break;
        case 2:
          EXPECT_NEAR(px[2], raw, 1e-6);
          break;
      }
    }
  }
#endif
}

TEST(MetalRawOpsTest, ClampTextureOnlyClampsUpperBound) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  const cv::Mat src = MakeClampedImage();

  metal::MetalImage image;
  image.Upload(src);

  ASSERT_NO_THROW(metal::utils::ClampTexture(image));

  cv::Mat clamped_gpu;
  image.Download(clamped_gpu);

  cv::Mat expected = src.clone();
  cv::min(expected, 1.0f, expected);

  ASSERT_EQ(clamped_gpu.type(), CV_32FC1);
  ASSERT_EQ(clamped_gpu.size(), expected.size());
  EXPECT_LE(cv::norm(clamped_gpu, expected, cv::NORM_INF), 1e-6);
#endif
}

TEST(MetalRawOpsTest, HighlightReconstructMatchesCPUReference) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);

  const cv::Mat input = MakeHighlightInput(48, 54);

  cv::Mat cpu_result = input.clone();
  ASSERT_NO_THROW(CPU::HighlightReconstruct(cpu_result, raw_processor));

  metal::MetalImage image;
  image.Upload(input);
  ASSERT_NO_THROW(metal::HighlightReconstruct(image, raw_processor));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  ASSERT_EQ(gpu_result.type(), CV_32FC1);
  ASSERT_EQ(gpu_result.size(), cpu_result.size());
  EXPECT_LE(cv::norm(gpu_result, cpu_result, cv::NORM_INF), 2e-5);
#endif
}

}  // namespace puerhlab
