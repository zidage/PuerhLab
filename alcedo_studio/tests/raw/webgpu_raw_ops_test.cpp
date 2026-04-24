//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "decoders/processor/operators/gpu/webgpu_to_linear_ref.hpp"
#include "decoders/processor/raw_normalization.hpp"
#include "decoders/processor/raw_processor_pattern.hpp"
#include "image/webgpu_image.hpp"
#include "webgpu/webgpu_context.hpp"

namespace alcedo {
namespace {

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

auto MakeXTransPattern() -> XTransPattern6x6 {
  static constexpr int kRawFc[36] = {
      1, 2, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 1,
      1, 2, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2, 2, 1, 2, 0, 1, 0,
  };

  XTransPattern6x6 pattern = {};
  for (int i = 0; i < 36; ++i) {
    pattern.raw_fc[i] = kRawFc[i];
    pattern.rgb_fc[i] = FoldRawColorToRgb(kRawFc[i]);
  }
  return pattern;
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
  raw_processor.imgdata.rawdata.color.maximum       = 15000;
  raw_processor.imgdata.rawdata.color.linear_max[0] = 14000;
  raw_processor.imgdata.rawdata.color.linear_max[1] = 14500;
  raw_processor.imgdata.rawdata.color.linear_max[2] = 14300;
  raw_processor.imgdata.rawdata.color.linear_max[3] = 14500;
  raw_processor.imgdata.rawdata.color.cam_mul[0]    = 2.4f;
  raw_processor.imgdata.rawdata.color.cam_mul[1]    = 1.0f;
  raw_processor.imgdata.rawdata.color.cam_mul[2]    = 1.7f;
  raw_processor.imgdata.rawdata.color.cam_mul[3]    = 1.0f;
  raw_processor.imgdata.rawdata.color.pre_mul[0]    = 2.1f;
  raw_processor.imgdata.rawdata.color.pre_mul[1]    = 1.0f;
  raw_processor.imgdata.rawdata.color.pre_mul[2]    = 1.6f;
  raw_processor.imgdata.rawdata.color.pre_mul[3]    = 1.0f;
  raw_processor.imgdata.color.as_shot_wb_applied    = 0;
}

auto MakeLinearizationRawProcessor() -> std::unique_ptr<LibRaw> {
  auto raw_processor = std::make_unique<LibRaw>();
  InitLinearizationRawProcessor(*raw_processor);
  return raw_processor;
}

auto ComputeLinearizedReference(const cv::Mat& raw_u16, const RawCfaPattern& pattern,
                                const LibRaw& raw_processor) -> cv::Mat {
  cv::Mat expected(raw_u16.rows, raw_u16.cols, CV_32FC1);
  const auto raw_curve = raw_norm::BuildLinearizationCurve(raw_processor.imgdata.rawdata);
  const bool apply_wb  = raw_processor.imgdata.color.as_shot_wb_applied != 1;

  for (int y = 0; y < raw_u16.rows; ++y) {
    for (int x = 0; x < raw_u16.cols; ++x) {
      const int   color  = RawColorAt(pattern, y, x);
      const float sample = static_cast<float>(raw_u16.at<uint16_t>(y, x));
      const float black  = raw_curve.black_level[color] +
                          raw_norm::PatternBlackAt(raw_processor.imgdata.rawdata, y, x);
      float pixel = raw_norm::NormalizeSample(sample, black, raw_curve.white_level[color]);
      pixel *= raw_norm::RelativeWhiteBalanceMultiplier(raw_processor.imgdata.rawdata.color.cam_mul,
                                                        color, apply_wb);
      expected.at<float>(y, x) = pixel;
    }
  }

  return expected;
}

auto WebGpuAvailable() -> bool {
  try {
    return webgpu::WebGpuContext::Instance().IsAvailable();
  } catch (const std::exception&) {
    return false;
  }
}

auto WebGpuInitializationLog() -> const std::string& {
  static const std::string unavailable = "WebGPU context initialization threw before logging.";
  try {
    return webgpu::WebGpuContext::Instance().InitializationLog();
  } catch (...) {
    return unavailable;
  }
}

}  // namespace

TEST(WebGpuRawOpsTest, ToLinearRefMatchesScalarReferenceForBayerAndXTrans) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n" << WebGpuInitializationLog();
  }

  auto raw_processor = MakeLinearizationRawProcessor();

  // --- Bayer test ---
  const BayerPattern2x2 bayer_pattern = MakePattern(0);
  RawCfaPattern         bayer_cfa     = {};
  bayer_cfa.kind                    = RawCfaKind::Bayer2x2;
  bayer_cfa.bayer_pattern           = bayer_pattern;

  cv::Mat bayer_raw(6, 8, CV_16UC1);
  for (int y = 0; y < bayer_raw.rows; ++y) {
    for (int x = 0; x < bayer_raw.cols; ++x) {
      bayer_raw.at<uint16_t>(y, x) = static_cast<uint16_t>(1500 + 47 * y + 31 * x);
    }
  }

  webgpu::WebGpuImage bayer_image;
  bayer_image.Upload(bayer_raw);
  ASSERT_NO_THROW(webgpu::ToLinearRef(bayer_image, *raw_processor, bayer_cfa));

  cv::Mat bayer_gpu;
  bayer_image.Download(bayer_gpu);
  const cv::Mat bayer_expected = ComputeLinearizedReference(bayer_raw, bayer_cfa, *raw_processor);
  EXPECT_LE(cv::norm(bayer_gpu, bayer_expected, cv::NORM_INF), 2e-5);

  // --- X-Trans test ---
  const XTransPattern6x6 xtrans_pattern = MakeXTransPattern();
  RawCfaPattern          xtrans_cfa     = {};
  xtrans_cfa.kind                     = RawCfaKind::XTrans6x6;
  xtrans_cfa.xtrans_pattern           = xtrans_pattern;

  cv::Mat xtrans_raw(8, 10, CV_16UC1);
  for (int y = 0; y < xtrans_raw.rows; ++y) {
    for (int x = 0; x < xtrans_raw.cols; ++x) {
      xtrans_raw.at<uint16_t>(y, x) = static_cast<uint16_t>(1800 + 29 * y + 19 * x);
    }
  }

  webgpu::WebGpuImage xtrans_image;
  xtrans_image.Upload(xtrans_raw);
  ASSERT_NO_THROW(webgpu::ToLinearRef(xtrans_image, *raw_processor, xtrans_cfa));

  cv::Mat xtrans_gpu;
  xtrans_image.Download(xtrans_gpu);
  const cv::Mat xtrans_expected =
      ComputeLinearizedReference(xtrans_raw, xtrans_cfa, *raw_processor);
  EXPECT_LE(cv::norm(xtrans_gpu, xtrans_expected, cv::NORM_INF), 2e-5);
#endif
}

TEST(WebGpuRawOpsTest, ToLinearRefRejectsNonR16UintInput) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n" << WebGpuInitializationLog();
  }

  auto raw_processor = MakeLinearizationRawProcessor();
  const auto pattern = MakePattern(0);
  RawCfaPattern cfa  = {};
  cfa.kind           = RawCfaKind::Bayer2x2;
  cfa.bayer_pattern  = pattern;

  cv::Mat float_img(4, 4, CV_32FC1, cv::Scalar(0.5f));
  webgpu::WebGpuImage image;
  image.Upload(float_img);

  EXPECT_THROW(webgpu::ToLinearRef(image, *raw_processor, cfa), std::runtime_error);
#endif
}

TEST(WebGpuRawOpsTest, ToLinearRefRejectsEmptyImage) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n" << WebGpuInitializationLog();
  }

  auto raw_processor = MakeLinearizationRawProcessor();
  const auto pattern = MakePattern(0);
  RawCfaPattern cfa  = {};
  cfa.kind           = RawCfaKind::Bayer2x2;
  cfa.bayer_pattern  = pattern;

  webgpu::WebGpuImage image;
  EXPECT_THROW(webgpu::ToLinearRef(image, *raw_processor, cfa), std::runtime_error);
#endif
}

}  // namespace alcedo
