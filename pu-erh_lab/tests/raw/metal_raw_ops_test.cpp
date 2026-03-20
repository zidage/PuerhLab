//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/raw_normalization.hpp"
#include "decoders/processor/operators/gpu/metal_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/metal_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/metal_to_linear_ref.hpp"
#include "decoders/processor/operators/gpu/metal_xtrans_interpolate.hpp"
#include "decoders/processor/raw_processor_pattern.hpp"
#include "image/metal_image.hpp"
#include "metal/metal_context.hpp"
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

auto CFAColorAt(const BayerPattern2x2& pattern, int y, int x) -> int {
  return pattern.rgb_fc[BayerCellIndex(y, x)];
}

auto CFAColorAt(const XTransPattern6x6& pattern, int y, int x) -> int {
  return RgbColorAt(pattern, y, x);
}

auto MakePattern(const int top_left_raw_color) -> BayerPattern2x2 {
  switch (top_left_raw_color) {
    case 0:
      return {{0, 1, 3, 2}, {0, 1, 1, 2}};
    case 1:
      return {{1, 0, 2, 3}, {1, 0, 2, 1}};
    case 2:
      return {{2, 3, 1, 0}, {2, 1, 1, 0}};
    case 3:
      return {{3, 2, 0, 1}, {1, 2, 0, 1}};
    default:
      throw std::runtime_error("unsupported test Bayer pattern");
  }
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

auto MakeXTransPattern() -> XTransPattern6x6 {
  static constexpr int kRawFc[36] = {
      1, 2, 1, 1, 0, 1,
      1, 0, 1, 2, 1, 2,
      0, 1, 0, 1, 2, 1,
      1, 2, 1, 1, 0, 1,
      1, 0, 1, 2, 1, 2,
      2, 1, 2, 0, 1, 0,
  };

  XTransPattern6x6 pattern = {};
  for (int i = 0; i < 36; ++i) {
    pattern.raw_fc[i] = kRawFc[i];
    pattern.rgb_fc[i] = FoldRawColorToRgb(kRawFc[i]);
  }
  return pattern;
}

auto MakeXTransRaw(int rows, int cols, const XTransPattern6x6& pattern) -> cv::Mat {
  cv::Mat raw(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    float* row = raw.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      static constexpr float kByColor[3] = {0.75f, 0.52f, 0.21f};
      row[x] = kByColor[CFAColorAt(pattern, y, x)] + 0.001f * float((7 * y + 3 * x) % 11);
    }
  }
  return raw;
}

auto MakeHighlightInput(int rows, int cols, const BayerPattern2x2& pattern) -> cv::Mat {
  cv::Mat img(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    float* row = img.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      static constexpr float kBaseByColor[3] = {0.9810f, 0.9845f, 0.9790f};
      const int              color           = CFAColorAt(pattern, y, x);
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

auto MetalRuntimeAvailable() -> bool {
  try {
    return MetalContext::Instance().Device() != nullptr;
  } catch (const std::exception&) {
    return false;
  }
}

void InitLinearizationRawProcessor(LibRaw& raw_processor) {
  raw_processor.imgdata.color.as_shot_wb_applied = 0;
  raw_processor.imgdata.rawdata.color.black      = 512;
  raw_processor.imgdata.rawdata.color.cblack[0]  = 16;
  raw_processor.imgdata.rawdata.color.cblack[1]  = 32;
  raw_processor.imgdata.rawdata.color.cblack[2]  = 48;
  raw_processor.imgdata.rawdata.color.cblack[3]  = 32;
  raw_processor.imgdata.rawdata.color.cblack[4]  = 6;
  raw_processor.imgdata.rawdata.color.cblack[5]  = 6;
  for (int i = 0; i < 36; ++i) {
    raw_processor.imgdata.rawdata.color.cblack[6 + i] = static_cast<unsigned short>(8 + (i % 5) * 3);
  }
  raw_processor.imgdata.rawdata.color.maximum    = 15000;
  raw_processor.imgdata.rawdata.color.linear_max[0] = 14000;
  raw_processor.imgdata.rawdata.color.linear_max[1] = 14500;
  raw_processor.imgdata.rawdata.color.linear_max[2] = 14300;
  raw_processor.imgdata.rawdata.color.linear_max[3] = 14500;
  raw_processor.imgdata.rawdata.color.cam_mul[0] = 2.4f;
  raw_processor.imgdata.rawdata.color.cam_mul[1] = 1.0f;
  raw_processor.imgdata.rawdata.color.cam_mul[2] = 1.7f;
  raw_processor.imgdata.rawdata.color.cam_mul[3] = 1.0f;
  raw_processor.imgdata.rawdata.color.pre_mul[0] = 2.1f;
  raw_processor.imgdata.rawdata.color.pre_mul[1] = 1.0f;
  raw_processor.imgdata.rawdata.color.pre_mul[2] = 1.6f;
  raw_processor.imgdata.rawdata.color.pre_mul[3] = 1.0f;
  raw_processor.imgdata.color.as_shot_wb_applied = 0;
}

auto ComputeLinearizedReference(const cv::Mat& raw_u16, const RawCfaPattern& pattern,
                                const LibRaw& raw_processor) -> cv::Mat {
  cv::Mat expected(raw_u16.rows, raw_u16.cols, CV_32FC1);
  const auto raw_curve = raw_norm::BuildLinearizationCurve(raw_processor.imgdata.rawdata);
  const bool apply_wb  = raw_processor.imgdata.color.as_shot_wb_applied != 1;

  for (int y = 0; y < raw_u16.rows; ++y) {
    for (int x = 0; x < raw_u16.cols; ++x) {
      const int color = RawColorAt(pattern, y, x);
      const float sample = static_cast<float>(raw_u16.at<uint16_t>(y, x));
      const float black =
          raw_curve.black_level[color] + raw_norm::PatternBlackAt(raw_processor.imgdata.rawdata, y, x);
      float pixel = raw_norm::NormalizeSample(sample, black, raw_curve.white_level[color]);
      pixel *= raw_norm::RelativeWhiteBalanceMultiplier(raw_processor.imgdata.rawdata.color.cam_mul,
                                                        color, apply_wb);
      expected.at<float>(y, x) = pixel;
    }
  }

  return expected;
}

}  // namespace

TEST(MetalRawOpsTest, CropRectMatchesCPUReference) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
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
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  const cv::Mat bayer = MakeBayerPattern(18, 20);
  const auto    pattern = MakePattern(1);

  metal::MetalImage image;
  image.Upload(bayer);

  ASSERT_NO_THROW(metal::Bayer2x2ToRGB_RCD(image, pattern));

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

      switch (CFAColorAt(pattern, y, x)) {
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
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
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
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);
  const auto pattern = MakePattern(0);

  const cv::Mat input = MakeHighlightInput(48, 54, pattern);

  cv::Mat cpu_result = input.clone();
  ASSERT_NO_THROW(CPU::HighlightReconstruct(cpu_result, raw_processor));

  metal::MetalImage image;
  image.Upload(input);
  ASSERT_NO_THROW(metal::HighlightReconstruct(image, raw_processor, pattern));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  ASSERT_EQ(gpu_result.type(), CV_32FC1);
  ASSERT_EQ(gpu_result.size(), cpu_result.size());
  EXPECT_LE(cv::norm(gpu_result, cpu_result, cv::NORM_INF), 2e-5);
#endif
}

TEST(MetalRawOpsTest, HighlightReconstructSupportsNonRggbBayerPatterns) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);
  const auto pattern = MakePattern(3);

  const cv::Mat input = MakeHighlightInput(48, 54, pattern);

  metal::MetalImage image;
  image.Upload(input);
  ASSERT_NO_THROW(metal::HighlightReconstruct(image, raw_processor, pattern));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  ASSERT_EQ(gpu_result.type(), CV_32FC1);
  ASSERT_EQ(gpu_result.size(), input.size());
  EXPECT_TRUE(cv::checkRange(gpu_result, true, nullptr, 0.0, 4.0));
#endif
}

TEST(MetalRawOpsTest, ToLinearRefMatchesScalarReferenceForBayerAndXTrans) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  LibRaw raw_processor;
  InitLinearizationRawProcessor(raw_processor);

  const BayerPattern2x2 bayer_pattern = MakePattern(0);
  RawCfaPattern         bayer_cfa = {};
  bayer_cfa.kind                    = RawCfaKind::Bayer2x2;
  bayer_cfa.bayer_pattern           = bayer_pattern;

  cv::Mat bayer_raw(6, 8, CV_16UC1);
  for (int y = 0; y < bayer_raw.rows; ++y) {
    for (int x = 0; x < bayer_raw.cols; ++x) {
      bayer_raw.at<uint16_t>(y, x) = static_cast<uint16_t>(1500 + 47 * y + 31 * x);
    }
  }

  metal::MetalImage bayer_image;
  bayer_image.Upload(bayer_raw);
  ASSERT_NO_THROW(metal::ToLinearRef(bayer_image, raw_processor, bayer_cfa));

  cv::Mat bayer_gpu;
  bayer_image.Download(bayer_gpu);
  const cv::Mat bayer_expected = ComputeLinearizedReference(bayer_raw, bayer_cfa, raw_processor);
  EXPECT_LE(cv::norm(bayer_gpu, bayer_expected, cv::NORM_INF), 2e-5);

  const XTransPattern6x6 xtrans_pattern = MakeXTransPattern();
  RawCfaPattern         xtrans_cfa = {};
  xtrans_cfa.kind                    = RawCfaKind::XTrans6x6;
  xtrans_cfa.xtrans_pattern          = xtrans_pattern;

  cv::Mat xtrans_raw(8, 10, CV_16UC1);
  for (int y = 0; y < xtrans_raw.rows; ++y) {
    for (int x = 0; x < xtrans_raw.cols; ++x) {
      xtrans_raw.at<uint16_t>(y, x) = static_cast<uint16_t>(1800 + 29 * y + 19 * x);
    }
  }

  metal::MetalImage xtrans_image;
  xtrans_image.Upload(xtrans_raw);
  ASSERT_NO_THROW(metal::ToLinearRef(xtrans_image, raw_processor, xtrans_cfa));

  cv::Mat xtrans_gpu;
  xtrans_image.Download(xtrans_gpu);
  const cv::Mat xtrans_expected = ComputeLinearizedReference(xtrans_raw, xtrans_cfa, raw_processor);
  EXPECT_LE(cv::norm(xtrans_gpu, xtrans_expected, cv::NORM_INF), 2e-5);
#endif
}

TEST(MetalRawOpsTest, XTransInterpolateProducesRGBAAndPreservesKnownSamples) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  if (!MetalRuntimeAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }
  const XTransPattern6x6 pattern = MakeXTransPattern();
  const cv::Mat raw = MakeXTransRaw(18, 18, pattern);

  metal::MetalImage image;
  image.Upload(raw);
  ASSERT_NO_THROW(metal::XTransToRGB_Ref(image, pattern, 1));

  cv::Mat rgba;
  image.Download(rgba);

  ASSERT_EQ(rgba.type(), CV_32FC4);
  ASSERT_EQ(rgba.size(), raw.size());
  EXPECT_TRUE(cv::checkRange(rgba, true, nullptr, 0.0, 4.0));

  for (int y = 0; y < rgba.rows; ++y) {
    const float* raw_row = raw.ptr<float>(y);
    const cv::Vec4f* rgba_row = rgba.ptr<cv::Vec4f>(y);
    for (int x = 0; x < rgba.cols; ++x) {
      const int color = CFAColorAt(pattern, y, x);
      EXPECT_NEAR(rgba_row[x][3], 1.0f, 1e-6);
      EXPECT_NEAR(rgba_row[x][color], raw_row[x], 1e-6);
    }
  }
#endif
}

}  // namespace puerhlab
