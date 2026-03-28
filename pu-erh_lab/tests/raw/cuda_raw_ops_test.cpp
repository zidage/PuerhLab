//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

#include <libraw/libraw.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"

namespace puerhlab {
namespace {

auto EnsureCudaDevice() -> bool {
  const int device_count = cv::cuda::getCudaEnabledDeviceCount();
  if (device_count <= 0) {
    return false;
  }
  cv::cuda::setDevice(0);
  return true;
}

void InitHighlightRawProcessor(LibRaw& raw_processor, const float red_mul = 2.2f,
                               const float green_mul = 1.0f, const float blue_mul = 1.55f) {
  raw_processor.imgdata.color.cam_mul[0] = red_mul;
  raw_processor.imgdata.color.cam_mul[1] = green_mul;
  raw_processor.imgdata.color.cam_mul[2] = blue_mul;
  raw_processor.imgdata.color.cam_mul[3] = 1.0f;
}

auto MakeRgbPlateau(int rows, int cols, const cv::Rect& plateau, bool fully_clipped) -> cv::Mat {
  cv::Mat img(rows, cols, CV_32FC3);
  for (int y = 0; y < rows; ++y) {
    cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
    for (int x = 0; x < cols; ++x) {
      const float fx = static_cast<float>(x) / static_cast<float>(std::max(cols - 1, 1));
      const float fy = static_cast<float>(y) / static_cast<float>(std::max(rows - 1, 1));
      cv::Vec3f   value(1.72f + 0.06f * fx, 0.79f + 0.04f * fy, 0.84f + 0.05f * fx);

      if (plateau.contains(cv::Point(x, y))) {
        if (fully_clipped) {
          value = cv::Vec3f(2.55f + 0.02f * float((x + y) & 1), 1.14f + 0.01f * float(x & 1),
                            1.72f + 0.02f * float(y & 1));
        } else {
          value = cv::Vec3f(2.45f + 0.05f * float(x & 1), 1.11f + 0.03f * float(y & 1),
                            0.54f + 0.01f * float((x + y) & 1));
        }
      }

      row[x] = value;
    }
  }
  return img;
}

auto MakePhasePatternPlateau(int rows, int cols, const cv::Rect& plateau) -> cv::Mat {
  cv::Mat img = MakeRgbPlateau(rows, cols, plateau, false);
  for (int y = plateau.y; y < plateau.y + plateau.height; ++y) {
    cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
    for (int x = plateau.x; x < plateau.x + plateau.width; ++x) {
      const int phase = ((y & 1) << 1) | (x & 1);
      row[x][0] += 0.18f * static_cast<float>(phase);
      row[x][1] += 0.08f * static_cast<float>(3 - phase);
      row[x][2] += 0.02f * static_cast<float>(phase & 1);
    }
  }
  return img;
}

auto PhaseSpread(const cv::Mat& image, const cv::Rect& roi, const int channel) -> float {
  std::array<double, 4> sums = {0.0, 0.0, 0.0, 0.0};
  std::array<int, 4>    cnts = {0, 0, 0, 0};

  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const cv::Vec3f* row = image.ptr<cv::Vec3f>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      const int phase = ((y & 1) << 1) | (x & 1);
      sums[phase] += row[x][channel];
      cnts[phase] += 1;
    }
  }

  float min_mean = std::numeric_limits<float>::max();
  float max_mean = std::numeric_limits<float>::lowest();
  for (int i = 0; i < 4; ++i) {
    if (cnts[i] == 0) {
      continue;
    }
    const float mean = static_cast<float>(sums[i] / static_cast<double>(cnts[i]));
    min_mean         = std::min(min_mean, mean);
    max_mean         = std::max(max_mean, mean);
  }

  return max_mean - min_mean;
}

auto ChannelSpread(const cv::Vec3f& pixel) -> float {
  return std::max(pixel[0], std::max(pixel[1], pixel[2])) -
         std::min(pixel[0], std::min(pixel[1], pixel[2]));
}

auto MaxDifferenceOutsideRoi(const cv::Mat& before, const cv::Mat& after, const cv::Rect& roi)
    -> float {
  float max_diff = 0.0f;
  for (int y = 0; y < before.rows; ++y) {
    const cv::Vec3f* before_row = before.ptr<cv::Vec3f>(y);
    const cv::Vec3f* after_row  = after.ptr<cv::Vec3f>(y);
    for (int x = 0; x < before.cols; ++x) {
      if (roi.contains(cv::Point(x, y))) {
        continue;
      }
      const cv::Vec3f delta = before_row[x] - after_row[x];
      max_diff =
          std::max(max_diff, std::max(std::abs(delta[0]), std::max(std::abs(delta[1]), std::abs(delta[2]))));
    }
  }
  return max_diff;
}

}  // namespace

TEST(CudaRawOpsTest, Clamp01SupportsRgb) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  cv::Mat src(4, 5, CV_32FC3);
  for (int y = 0; y < src.rows; ++y) {
    cv::Vec3f* row = src.ptr<cv::Vec3f>(y);
    for (int x = 0; x < src.cols; ++x) {
      row[x] = cv::Vec3f(-0.35f + 0.2f * x, 0.15f * y, 0.85f + 0.18f * float(x + y));
    }
  }

  cv::cuda::GpuMat gpu(src);
  ASSERT_NO_THROW(CUDA::Clamp01(gpu));

  cv::Mat clamped;
  gpu.download(clamped);
  ASSERT_EQ(clamped.type(), CV_32FC3);

  for (int y = 0; y < src.rows; ++y) {
    const cv::Vec3f* row = clamped.ptr<cv::Vec3f>(y);
    for (int x = 0; x < src.cols; ++x) {
      EXPECT_GE(row[x][0], 0.0f);
      EXPECT_GE(row[x][1], 0.0f);
      EXPECT_GE(row[x][2], 0.0f);
      EXPECT_LE(row[x][0], 1.0f);
      EXPECT_LE(row[x][1], 1.0f);
      EXPECT_LE(row[x][2], 1.0f);
    }
  }
#endif
}

TEST(CudaRawOpsTest, HighlightReconstructNeutralizesFullyClippedHighlights) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);
  const cv::Rect plateau(12, 10, 20, 18);
  const cv::Mat  input = MakeRgbPlateau(40, 44, plateau, true);

  cv::cuda::GpuMat gpu(input);
  ASSERT_NO_THROW(CUDA::HighlightReconstruct(gpu, raw_processor));

  cv::Mat output;
  gpu.download(output);
  ASSERT_EQ(output.type(), CV_32FC3);

  const cv::Vec3f center = output.at<cv::Vec3f>(plateau.y + plateau.height / 2,
                                                plateau.x + plateau.width / 2);
  EXPECT_LT(std::abs(center[0] - center[1]), 0.05f);
  EXPECT_LT(std::abs(center[1] - center[2]), 0.05f);
  EXPECT_LT(std::abs(center[0] - center[2]), 0.05f);
#endif
}

TEST(CudaRawOpsTest, HighlightReconstructDesaturatesTwoChannelClip) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);
  const cv::Rect plateau(10, 11, 22, 20);
  const cv::Mat  input = MakeRgbPlateau(42, 46, plateau, false);

  cv::cuda::GpuMat gpu(input);
  ASSERT_NO_THROW(CUDA::HighlightReconstruct(gpu, raw_processor));

  cv::Mat output;
  gpu.download(output);

  const cv::Vec3f before = input.at<cv::Vec3f>(plateau.y + plateau.height / 2,
                                               plateau.x + plateau.width / 2);
  const cv::Vec3f after =
      output.at<cv::Vec3f>(plateau.y + plateau.height / 2, plateau.x + plateau.width / 2);

  EXPECT_LT(ChannelSpread(after), ChannelSpread(before));
  EXPECT_LT(std::abs(after[1] - after[2]), std::abs(before[1] - before[2]));
#endif
}

TEST(CudaRawOpsTest, HighlightReconstructSuppressesPhaseLikePlateauPattern) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);
  const cv::Rect plateau(12, 12, 24, 24);
  const cv::Mat  input = MakePhasePatternPlateau(48, 52, plateau);

  const float before_spread = PhaseSpread(input, plateau, 1);

  cv::cuda::GpuMat gpu(input);
  ASSERT_NO_THROW(CUDA::HighlightReconstruct(gpu, raw_processor));

  cv::Mat output;
  gpu.download(output);

  const float after_spread = PhaseSpread(output, plateau, 1);
  EXPECT_LT(after_spread, before_spread * 0.35f);
#endif
}

TEST(CudaRawOpsTest, HighlightReconstructDoesNotBleedOutsideClippedRegion) {
#ifndef HAVE_CUDA
  GTEST_SKIP() << "CUDA is not enabled in this build.";
#else
  if (!EnsureCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable in this environment.";
  }

  LibRaw raw_processor;
  InitHighlightRawProcessor(raw_processor);
  const cv::Rect plateau(14, 13, 16, 15);
  const cv::Mat  input = MakeRgbPlateau(44, 47, plateau, false);

  cv::cuda::GpuMat gpu(input);
  ASSERT_NO_THROW(CUDA::HighlightReconstruct(gpu, raw_processor));

  cv::Mat output;
  gpu.download(output);

  EXPECT_LT(MaxDifferenceOutsideRoi(input, output, plateau), 1e-6f);
#endif
}

}  // namespace puerhlab
