//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QTimer>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

#include "decoders/libraw_unpack_guard.hpp"
#include "decoders/processor/operators/gpu/webgpu_cvt_ref_space.hpp"
#include "decoders/processor/operators/gpu/webgpu_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/webgpu_to_linear_ref.hpp"
#include "decoders/processor/raw_normalization.hpp"
#include "decoders/processor/raw_processor_pattern.hpp"
#include "image/webgpu_image.hpp"
#include "webgpu/webgpu_context.hpp"
#include "webgpu/webgpu_geometry_utils.hpp"

namespace alcedo {
namespace {
using Clock = std::chrono::steady_clock;

auto MsSince(const Clock::time_point start) -> double {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

auto OpenRawFile(LibRaw& raw_processor, const std::filesystem::path& path) -> int {
#ifdef _WIN32
  return raw_processor.open_file(path.wstring().c_str());
#else
  return raw_processor.open_file(path.string().c_str());
#endif
}

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

auto ApplyInverseCamMulReference(const cv::Mat& src, const std::array<float, 4>& cam_mul)
    -> cv::Mat {
  const float g      = std::max(cam_mul[1], 1e-6f);
  const float gain_r = g / std::max(cam_mul[0], 1e-6f);
  const float gain_b = g / std::max(cam_mul[2], 1e-6f);

  cv::Mat     scaled(src.size(), CV_32FC4);
  for (int y = 0; y < src.rows; ++y) {
    const cv::Vec4f* src_row = src.ptr<cv::Vec4f>(y);
    cv::Vec4f*       dst_row = scaled.ptr<cv::Vec4f>(y);
    for (int x = 0; x < src.cols; ++x) {
      dst_row[x] = cv::Vec4f(src_row[x][0] * gain_r, src_row[x][1], src_row[x][2] * gain_b, 1.0f);
    }
  }
  return scaled;
}

auto CFAColorAt(const BayerPattern2x2& pattern, int y, int x) -> int {
  return pattern.rgb_fc[BayerCellIndex(y, x)];
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
    raw_processor.imgdata.rawdata.color.cblack[6 + i] =
        static_cast<unsigned short>(8 + (i % 5) * 3);
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
  cv::Mat    expected(raw_u16.rows, raw_u16.cols, CV_32FC1);
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

auto LeicaM10RawPath() -> std::filesystem::path {
  return std::filesystem::path(TEST_IMG_PATH) / "raw" / "camera" / "leica" / "m10" / "L1001108.dng";
}

auto RawPerformanceTestPath() -> std::filesystem::path {
  return "D:/Projects/pu-erh_lab/alcedo_studio/tests/resources/sample_images/raw/camera/lumix/s5/"
         "P1000625.RW2";
}

auto MakeDisplayPreview(const cv::Mat& rgba_linear) -> QImage {
  std::vector<cv::Mat> channels;
  cv::split(rgba_linear, channels);
  cv::Mat rgb;
  cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, rgb);

  cv::Mat normalized_rgb;
  cv::normalize(rgb, normalized_rgb, 0.0, 1.0, cv::NORM_MINMAX);

  std::vector<cv::Mat> normalized_channels;
  cv::split(normalized_rgb, normalized_channels);
  normalized_channels.push_back(cv::Mat(rgba_linear.size(), CV_32FC1, cv::Scalar(1.0f)));

  cv::Mat normalized_rgba;
  cv::merge(normalized_channels, normalized_rgba);

  cv::Mat rgba8;
  normalized_rgba.convertTo(rgba8, CV_8UC4, 255.0);
  if (!rgba8.isContinuous()) {
    rgba8 = rgba8.clone();
  }

  QImage image(rgba8.data, rgba8.cols, rgba8.rows, static_cast<int>(rgba8.step),
               QImage::Format_RGBA8888);
  return image.copy();
}

void PrintMatStats(const char* label, const cv::Mat& mat) {
  std::vector<cv::Mat> channels;
  cv::split(mat, channels);
  std::cout << "[WebGPU RAW stats] " << label << " type=" << mat.type() << " size=" << mat.cols
            << 'x' << mat.rows;
  for (size_t c = 0; c < channels.size(); ++c) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(channels[c], &min_value, &max_value);
    const cv::Scalar mean = cv::mean(channels[c]);
    std::cout << " c" << c << "{min=" << min_value << ", max=" << max_value << ", mean=" << mean[0]
              << '}';
  }
  std::cout << '\n';
}

auto DownsampleRawForPreview(const cv::Mat& raw, RawCfaPattern& pattern, const int passes)
    -> cv::Mat {
  if (passes <= 0) {
    return raw.clone();
  }

  cv::Mat downsampled = DownsampleRaw2x(raw, pattern);
  for (int pass = 1; pass < passes; ++pass) {
    downsampled = DownsampleRaw2x(downsampled, pattern);
  }
  return downsampled;
}

auto EnsureQApplication() -> QApplication* {
  if (auto* app = qobject_cast<QApplication*>(QCoreApplication::instance())) {
    return app;
  }

  static int   argc       = 1;
  static char  app_name[] = "WebGpuRawOpsTest";
  static char* argv[]     = {app_name, nullptr};
  static auto  app        = std::make_unique<QApplication>(argc, argv);
  return app.get();
}

void ShowPreviewWithQt(const QImage& preview, const QString& title) {
  ASSERT_FALSE(preview.isNull());

  auto* app = EnsureQApplication();
  if (app == nullptr) {
    GTEST_SKIP() << "QApplication could not be initialized for Qt preview.";
    return;
  }

  QLabel label;
  label.setWindowTitle(title);
  label.setAlignment(Qt::AlignCenter);
  label.setPixmap(
      QPixmap::fromImage(preview).scaled(1280, 900, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  label.resize(label.pixmap().size());
  label.show();

  QTimer::singleShot(150000, app, &QCoreApplication::quit);
  app->exec();
}

}  // namespace

TEST(WebGpuRawOpsTest, ToLinearRefMatchesScalarReferenceForBayerAndXTrans) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  auto                  raw_processor = MakeLinearizationRawProcessor();

  // --- Bayer test ---
  const BayerPattern2x2 bayer_pattern = MakePattern(0);
  RawCfaPattern         bayer_cfa     = {};
  bayer_cfa.kind                      = RawCfaKind::Bayer2x2;
  bayer_cfa.bayer_pattern             = bayer_pattern;

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
  xtrans_cfa.kind                       = RawCfaKind::XTrans6x6;
  xtrans_cfa.xtrans_pattern             = xtrans_pattern;

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
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  auto          raw_processor = MakeLinearizationRawProcessor();
  const auto    pattern       = MakePattern(0);
  RawCfaPattern cfa           = {};
  cfa.kind                    = RawCfaKind::Bayer2x2;
  cfa.bayer_pattern           = pattern;

  cv::Mat             float_img(4, 4, CV_32FC1, cv::Scalar(0.5f));
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
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  auto          raw_processor = MakeLinearizationRawProcessor();
  const auto    pattern       = MakePattern(0);
  RawCfaPattern cfa           = {};
  cfa.kind                    = RawCfaKind::Bayer2x2;
  cfa.bayer_pattern           = pattern;

  webgpu::WebGpuImage image;
  EXPECT_THROW(webgpu::ToLinearRef(image, *raw_processor, cfa), std::runtime_error);
#endif
}

TEST(WebGpuRawOpsTest, DebayerRcdProducesRGBAAndPreservesCFASamples) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const cv::Mat       bayer   = MakeBayerPattern(18, 20);
  const auto          pattern = MakePattern(1);

  webgpu::WebGpuImage image;
  image.Upload(bayer);

  ASSERT_NO_THROW(webgpu::Bayer2x2ToRGB_RCD(image, pattern));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  ASSERT_EQ(gpu_result.type(), CV_32FC4);
  ASSERT_EQ(gpu_result.size(), bayer.size());

  for (int y = 0; y < gpu_result.rows; ++y) {
    const float*     raw_row  = bayer.ptr<float>(y);
    const cv::Vec4f* rgba_row = gpu_result.ptr<cv::Vec4f>(y);
    for (int x = 0; x < gpu_result.cols; ++x) {
      const cv::Vec4f px  = rgba_row[x];
      const float     raw = raw_row[x];

      EXPECT_GE(px[0], 0.0f);
      EXPECT_GE(px[1], 0.0f);
      EXPECT_GE(px[2], 0.0f);
      EXPECT_NEAR(px[3], 1.0f, 1e-6);

      switch (CFAColorAt(pattern, y, x)) {
        case 0:
          EXPECT_NEAR(px[0], raw, 1e-5);
          break;
        case 1:
          EXPECT_NEAR(px[1], raw, 1e-5);
          break;
        case 2:
          EXPECT_NEAR(px[2], raw, 1e-5);
          break;
      }
    }
  }
#endif
}

TEST(WebGpuRawOpsTest, Clamp01ClampsSingleChannelRawTexture) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  cv::Mat src(3, 5, CV_32FC1);
  for (int y = 0; y < src.rows; ++y) {
    float* row = src.ptr<float>(y);
    for (int x = 0; x < src.cols; ++x) {
      row[x] = -0.5f + static_cast<float>(y * src.cols + x) * 0.2f;
    }
  }

  webgpu::WebGpuImage image;
  image.Upload(src);

  ASSERT_NO_THROW(webgpu::Clamp01(image));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  cv::Mat expected;
  cv::max(src, 0.0f, expected);
  cv::min(expected, 1.0f, expected);
  EXPECT_LE(cv::norm(gpu_result, expected, cv::NORM_INF), 1e-6);
#endif
}

TEST(WebGpuRawOpsTest, ApplyInverseCamMulAndOrientRGBAMatchesCpuReference) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const cv::Mat              src     = MakeRGBAImage(4, 7);
  const std::array<float, 4> cam_mul = {2.0f, 1.0f, 4.0f, 1.0f};

  webgpu::WebGpuImage        image;
  image.Upload(src);

  ASSERT_NO_THROW(webgpu::ApplyInverseCamMulAndOrientRGBA(image, cam_mul.data(), 6));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  const cv::Mat scaled = ApplyInverseCamMulReference(src, cam_mul);
  cv::Mat       expected;
  cv::transpose(scaled, expected);
  cv::flip(expected, expected, 1);

  ASSERT_EQ(gpu_result.type(), CV_32FC4);
  ASSERT_EQ(gpu_result.size(), expected.size());
  EXPECT_LE(cv::norm(gpu_result, expected, cv::NORM_INF), 1e-6);
#endif
}

TEST(WebGpuRawOpsTest, PreviewLeicaM10FullWebGpuRawPipelineWithQt) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const std::filesystem::path raw_path = RawPerformanceTestPath();
  if (!std::filesystem::exists(raw_path)) {
    GTEST_SKIP() << "Sample RAW file is missing: " << raw_path.string();
  }

  std::unique_ptr<LibRaw> raw_processor = std::make_unique<LibRaw>();

  auto                    stage_start   = Clock::now();
  int                     ret           = OpenRawFile(*raw_processor, raw_path);
  ASSERT_EQ(ret, LIBRAW_SUCCESS) << libraw_strerror(ret);
  const double open_ms = MsSince(stage_start);

  stage_start          = Clock::now();
  ret                  = libraw_guard::Unpack(*raw_processor);
  ASSERT_EQ(ret, LIBRAW_SUCCESS) << libraw_strerror(ret);
  const double unpack_ms = MsSince(stage_start);

  ASSERT_NE(raw_processor->imgdata.rawdata.raw_image, nullptr);
  RawCfaPattern cfa_pattern = ReadLibRawCfaPattern(*raw_processor);
  ASSERT_EQ(cfa_pattern.kind, RawCfaKind::Bayer2x2)
      << "This preview test currently expects a Bayer 2x2 RAW.";

  cv::Mat raw_view{static_cast<int>(raw_processor->imgdata.sizes.raw_height),
                   static_cast<int>(raw_processor->imgdata.sizes.raw_width), CV_16UC1,
                   raw_processor->imgdata.rawdata.raw_image};
  cv::Mat preview_raw = raw_view;

  stage_start         = Clock::now();
  webgpu::WebGpuImage image;
  image.Upload(preview_raw);
  const double upload_ms = MsSince(stage_start);

  stage_start            = Clock::now();
  webgpu::ToLinearRef(image, *raw_processor, cfa_pattern);
  const double linear_ms = MsSince(stage_start);

  cv::Mat      linear_result;
  image.Download(linear_result);
  PrintMatStats("linear", linear_result);

  stage_start = Clock::now();
  webgpu::Clamp01(image);
  const double clamp_ms = MsSince(stage_start);

  cv::Mat      clamped_result;
  image.Download(clamped_result);
  PrintMatStats("clamped", clamped_result);

  stage_start = Clock::now();
  webgpu::Bayer2x2ToRGB_RCD(image, cfa_pattern.bayer_pattern);
  const double debayer_ms = MsSince(stage_start);

  stage_start             = Clock::now();
  cv::Mat debayer_result;
  image.Download(debayer_result);
  const double debayer_download_ms = MsSince(stage_start);
  ASSERT_FALSE(debayer_result.empty());
  ASSERT_EQ(debayer_result.type(), CV_32FC4);
  ASSERT_EQ(debayer_result.cols, preview_raw.cols);
  ASSERT_EQ(debayer_result.rows, preview_raw.rows);
  PrintMatStats("debayer", debayer_result);

  ShowPreviewWithQt(MakeDisplayPreview(debayer_result),
                    QStringLiteral("WebGPU Leica M10 RCD debayer preview"));

  stage_start = Clock::now();
  webgpu::ApplyInverseCamMulAndOrientRGBA(image, raw_processor->imgdata.rawdata.color.cam_mul,
                                          raw_processor->imgdata.sizes.flip);
  const double inverse_wb_orient_ms = MsSince(stage_start);

  stage_start                       = Clock::now();
  cv::Mat full_result;
  image.Download(full_result);
  const double full_download_ms = MsSince(stage_start);
  ASSERT_FALSE(full_result.empty());
  ASSERT_EQ(full_result.type(), CV_32FC4);
  if (raw_processor->imgdata.sizes.flip == 5 || raw_processor->imgdata.sizes.flip == 6) {
    ASSERT_EQ(full_result.cols, preview_raw.rows);
    ASSERT_EQ(full_result.rows, preview_raw.cols);
  } else {
    ASSERT_EQ(full_result.cols, preview_raw.cols);
    ASSERT_EQ(full_result.rows, preview_raw.rows);
  }
  PrintMatStats("final", full_result);

  std::cout << "[WebGPU RAW preview] file=" << raw_path.string() << '\n'
            << "[WebGPU RAW preview] raw_size=" << raw_view.cols << 'x' << raw_view.rows
            << " preview_size=" << preview_raw.cols << 'x' << preview_raw.rows << '\n'
            << "[WebGPU RAW preview] open=" << open_ms << " ms"
            << " unpack=" << unpack_ms << " ms"
            << " upload=" << upload_ms << " ms"
            << " to_linear_ref=" << linear_ms << " ms"
            << " clamp=" << clamp_ms << " ms"
            << " debayer_rcd=" << debayer_ms << " ms"
            << " debayer_download=" << debayer_download_ms << " ms"
            << " inverse_wb_orient_rgba=" << inverse_wb_orient_ms << " ms"
            << " full_download=" << full_download_ms << " ms"
            << " gpu_ops_total=" << (linear_ms + clamp_ms + debayer_ms + inverse_wb_orient_ms)
            << " ms\n";

#endif
}

TEST(WebGpuRawOpsTest, MeasureLeicaM10FullWebGpuRawDecodePerformance) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const std::filesystem::path raw_path = RawPerformanceTestPath();
  if (!std::filesystem::exists(raw_path)) {
    GTEST_SKIP() << "Sample RAW file is missing: " << raw_path.string();
  }

  std::unique_ptr<LibRaw> raw_processor = std::make_unique<LibRaw>();

  const auto              total_start   = Clock::now();
  auto                    stage_start   = Clock::now();
  int                     ret           = OpenRawFile(*raw_processor, raw_path);
  ASSERT_EQ(ret, LIBRAW_SUCCESS) << libraw_strerror(ret);
  const double open_ms = MsSince(stage_start);

  stage_start          = Clock::now();
  ret                  = libraw_guard::Unpack(*raw_processor);
  ASSERT_EQ(ret, LIBRAW_SUCCESS) << libraw_strerror(ret);
  const double unpack_ms = MsSince(stage_start);

  ASSERT_NE(raw_processor->imgdata.rawdata.raw_image, nullptr);
  RawCfaPattern cfa_pattern = ReadLibRawCfaPattern(*raw_processor);
  ASSERT_EQ(cfa_pattern.kind, RawCfaKind::Bayer2x2)
      << "This performance test currently expects a Bayer 2x2 RAW.";

  cv::Mat raw_view{static_cast<int>(raw_processor->imgdata.sizes.raw_height),
                   static_cast<int>(raw_processor->imgdata.sizes.raw_width), CV_16UC1,
                   raw_processor->imgdata.rawdata.raw_image};

  stage_start = Clock::now();
  webgpu::WebGpuImage image;
  image.Upload(raw_view);
  const double upload_ms = MsSince(stage_start);

  stage_start            = Clock::now();
  webgpu::ToLinearRef(image, *raw_processor, cfa_pattern);
  const double linear_ms = MsSince(stage_start);

  stage_start            = Clock::now();
  webgpu::Clamp01(image);
  const double clamp_ms = MsSince(stage_start);

  stage_start           = Clock::now();
  webgpu::Bayer2x2ToRGB_RCD(image, cfa_pattern.bayer_pattern);
  const double debayer_ms = MsSince(stage_start);

  stage_start             = Clock::now();
  webgpu::ApplyInverseCamMulAndOrientRGBA(image, raw_processor->imgdata.rawdata.color.cam_mul,
                                          raw_processor->imgdata.sizes.flip);
  const double inverse_wb_orient_ms = MsSince(stage_start);
  const double total_ms             = MsSince(total_start);

  std::cout << "[WebGPU RAW perf] file=" << raw_path.string() << '\n'
            << "[WebGPU RAW perf] raw_size=" << raw_view.cols << 'x' << raw_view.rows << '\n'
            << "[WebGPU RAW perf] open=" << open_ms << " ms"
            << " unpack=" << unpack_ms << " ms"
            << " upload=" << upload_ms << " ms"
            << " to_linear_ref=" << linear_ms << " ms"
            << " clamp=" << clamp_ms << " ms"
            << " debayer_rcd=" << debayer_ms << " ms"
            << " inverse_wb_orient_rgba=" << inverse_wb_orient_ms << " ms"
            << " gpu_decode_total=" << (linear_ms + clamp_ms + debayer_ms + inverse_wb_orient_ms)
            << " ms"
            << " end_to_end_total=" << total_ms << " ms\n";
#endif
}

TEST(WebGpuRawOpsTest, Rotate180MatchesCpuReference) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const cv::Mat       src = MakeRGBAImage(9, 11);

  webgpu::WebGpuImage image;
  image.Upload(src);

  ASSERT_NO_THROW(webgpu::utils::Rotate180(image));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  cv::Mat expected;
  cv::flip(src, expected, -1);

  ASSERT_EQ(gpu_result.type(), CV_32FC4);
  ASSERT_EQ(gpu_result.size(), expected.size());
  EXPECT_LE(cv::norm(gpu_result, expected, cv::NORM_INF), 1e-6);
#endif
}

TEST(WebGpuRawOpsTest, Rotate90CWMatchesCpuReference) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const cv::Mat       src = MakeRGBAImage(9, 11);

  webgpu::WebGpuImage image;
  image.Upload(src);

  ASSERT_NO_THROW(webgpu::utils::Rotate90CW(image));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  cv::Mat expected;
  cv::transpose(src, expected);
  cv::flip(expected, expected, 1);

  ASSERT_EQ(gpu_result.type(), CV_32FC4);
  ASSERT_EQ(gpu_result.size(), expected.size());
  EXPECT_LE(cv::norm(gpu_result, expected, cv::NORM_INF), 1e-6);
#endif
}

TEST(WebGpuRawOpsTest, Rotate90CCWMatchesCpuReference) {
#ifndef HAVE_WEBGPU
  GTEST_SKIP() << "WebGPU is not enabled in this build.";
#else
  SCOPED_TRACE(WebGpuInitializationLog());
  if (!WebGpuAvailable()) {
    GTEST_SKIP() << "WebGPU device is unavailable in this environment.\n"
                 << WebGpuInitializationLog();
  }

  const cv::Mat       src = MakeRGBAImage(9, 11);

  webgpu::WebGpuImage image;
  image.Upload(src);

  ASSERT_NO_THROW(webgpu::utils::Rotate90CCW(image));

  cv::Mat gpu_result;
  image.Download(gpu_result);

  cv::Mat expected;
  cv::transpose(src, expected);
  cv::flip(expected, expected, 0);

  ASSERT_EQ(gpu_result.type(), CV_32FC4);
  ASSERT_EQ(gpu_result.size(), expected.size());
  EXPECT_LE(cv::norm(gpu_result, expected, cv::NORM_INF), 1e-6);
#endif
}

}  // namespace alcedo
